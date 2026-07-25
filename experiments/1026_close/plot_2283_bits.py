#!/usr/bin/env python3
"""Plot the #2283 paired Eq-4 bits-at-R2 rows against the theorem's prediction.

Two panels, both read from the SAME authoritative JSONL the comparator accepts:

* **left** — bits/token at each fixed-distortion R2 operating point for the
  external TopK bar and the theorem-faithful hybrid, with the CROSSOVER
  THEOREM's closed-form prediction for the hybrid drawn alongside. The
  prediction is a-priori: it takes the external row's measured code and residual
  terms (the theorem says a circle-class chart moves neither: the code delta
  ``(s-d-1)/2*log2(lambda/delta)`` vanishes at ``s = d+1 = 2`` and the
  matched-recon residual delta is zero) and swaps ONLY the support term, which
  the two configurations fix in closed form as ``log2 C(G, L0)``. The dictionary
  term cannot move at all: the faithful ``k_flat`` config equalises the decoder
  scalar counts exactly.
* **right** — the four-term decomposition at the acceptance target, so the
  reader can see which term the contest is actually decided on.

The hybrid's full-linear-span bracket (from each row's own faithfulness audit)
is drawn as the pessimistic end of the measurement: the score with every atom
charged its whole decoder span rather than the ``d+1`` scalars the theorem's
ledger names.
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from crossover_theorem_check import selection_bits  # noqa: E402

TARGETS = (0.80, 0.90, 0.95, 0.99)
ACCEPTANCE_TARGET = 0.99


def _rows(path: str, run_id: str) -> dict:
    found: dict[str, dict] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("run_id") != run_id or record.get("record_type") != "result":
                continue
            found[record["arm"]] = record
    missing = {"external_topk", "hybrid_rust"} - set(found)
    if missing:
        raise SystemExit(f"{path}: run {run_id!r} is missing rows for {sorted(missing)}")
    return found


def _series(record: dict, key: str) -> list[float]:
    return [float(record[f"bits_{key}_at_r2_{target:g}"]) for target in TARGETS]


def _theorem_support_bits(record: dict) -> tuple[float, float]:
    """Closed-form support cost of the two CONFIGURATIONS (not of the fits).

    External: ``G = K`` atoms, ``L0 = top_k`` actives. Hybrid: the faithful
    config replaces ``curved_atoms`` flat atoms' worth of decoder rows with
    charts, and each chart firing spends ``1 + d`` of the active-scalar budget
    where a flat firing spends one -- so the hybrid names fewer atoms per token.
    """
    k_external = int(record["K"])
    top_k = int(record["top_k"])
    curved_atoms = int(record["curved_atoms"])
    curved_k = int(record["curved_k"])
    d_atom = int(record["d_atom"])
    k_flat = int(record["k_flat"])
    flat_actives = top_k - curved_k * (1 + d_atom)
    external = selection_bits(k_external, top_k)
    hybrid = selection_bits(k_flat + curved_atoms, flat_actives + curved_k)
    return external, hybrid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = _rows(args.results, args.run_id)
    external, hybrid = rows["external_topk"], rows["hybrid_rust"]
    external_bits = _series(external, "bits")
    hybrid_bits = _series(hybrid, "bits")
    support_external, support_hybrid = _theorem_support_bits(hybrid)
    predicted = [value - (support_external - support_hybrid) for value in external_bits]

    audit = hybrid.get("bits_faithfulness_audit", {}).get("bracket", {})
    full_span = audit.get("full_span_bits")

    figure, (curve, split) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    curve.plot(TARGETS, external_bits, "o-", color="#B4413C", linewidth=2,
               label=f"external TopK (K={external['K']}), measured")
    curve.plot(TARGETS, hybrid_bits, "s-", color="#1F5C8B", linewidth=2,
               label=f"theorem-faithful hybrid (k_flat={hybrid['k_flat']}"
                     f"+{hybrid['curved_atoms']} charts), measured")
    curve.plot(TARGETS, predicted, "--", color="#1F5C8B", linewidth=1.4, alpha=0.75,
               label="crossover theorem prediction for the hybrid")
    if full_span is not None:
        curve.plot([ACCEPTANCE_TARGET], [full_span], "v", color="#1F5C8B",
                   markersize=9, markerfacecolor="none",
                   label="hybrid, every atom charged its full decoder span")
    curve.set_xlabel("fixed-distortion operating point  $R^2$")
    curve.set_ylabel("Eq-4 description length (bits / token)")
    curve.set_title(
        f"creditscope L30 residual_post, N={external['N']}, p={external['p']}, "
        f"horizon={external['amortization_horizon']}")
    curve.grid(alpha=0.25)
    curve.legend(fontsize=8, loc="upper left")

    names = ("dictionary", "support", "code", "residual")
    def parts(record):
        return [
            float(record["bits_dictionary_bits"]),
            float(record["bits_support_bits"]),
            float(record[f"bits_code_bits_at_r2_{ACCEPTANCE_TARGET:g}"]),
            float(record[f"bits_resid_bits_at_r2_{ACCEPTANCE_TARGET:g}"]),
        ]

    external_parts, hybrid_parts = parts(external), parts(hybrid)
    positions = range(len(names))
    width = 0.38
    split.bar([p - width / 2 for p in positions], external_parts, width,
              color="#B4413C", label="external TopK")
    split.bar([p + width / 2 for p in positions], hybrid_parts, width,
              color="#1F5C8B", label="hybrid")
    for index, (left, right) in enumerate(zip(external_parts, hybrid_parts, strict=True)):
        split.annotate(f"{left - right:+.1f}", (index, max(left, right)),
                       textcoords="offset points", xytext=(0, 4),
                       ha="center", fontsize=9)
    split.set_xticks(list(positions))
    split.set_xticklabels(names)
    split.set_yscale("symlog")
    split.set_ylabel(f"bits / token at $R^2$={ACCEPTANCE_TARGET}")
    split.set_title("term-by-term (annotation = external - hybrid)")
    split.grid(alpha=0.25, axis="y")
    split.legend(fontsize=9)

    figure.suptitle(
        f"gam #2283 - authoritative Eq-4 bits at $R^2$ = {ACCEPTANCE_TARGET}: "
        f"external {external_bits[-1]:.1f} vs hybrid {hybrid_bits[-1]:.1f} bits/token",
        fontsize=12)
    figure.tight_layout()
    figure.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
