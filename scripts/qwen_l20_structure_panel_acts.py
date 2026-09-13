#!/usr/bin/env python3
"""#2263 gate 5: the committed layer-20 weekday/month activation panel on Qwen3-8B.

Gate 5 names the historical L20 structure-certificate panel (0 of 39 fits minted),
whose activation set was never committed. This is its committed restatement: the
same model and layer, a committed prompt bank, and slices in the form
``experiments/scale_close/panel_2263_structure_certificate.py --acts-dir`` reads.

Each feature's cloud is every committed head followed by every label. The model is
causal, so the residual at the label token depends on the head alone, and distinct
heads are the cloud's only source of diversity (the design of E1's prompt bank).
One label-position residual row per head and label is captured at the hook layer
with float32 weights. The raw rows are saved as ``{feature}_L{layer}_raw.npz`` with
their head and label indices. The panel slice ``{feature}_L{layer}.npy`` is their
``chart_dim``-dimensional PCA chart, so the dense certification lane fits a bounded
width. ``panel_acts.json`` records the model revision, a content-addressed model
sha256, the prompt-bank sha256, and every chart's explained variance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative: str) -> Any:
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LEDGER = _load_module("qwen_l17_dose_ledger", "scripts/qwen_l17_dose_ledger.py")
MATRIX = LEDGER.MATRIX
CALENDAR = LEDGER.CALENDAR
E1 = LEDGER.E1
log = LEDGER.log


PANEL_PROTOCOL: dict[str, Any] = {
    "model": "Qwen/Qwen3-8B",
    "layer": 20,
    "model_dtype": "float32",
    "features": ("weekday", "month"),
    "chart_dim": 32,
}


MONTH_HEADS = (
    "The festival takes place in", "Her birthday falls in", "The school year starts in",
    "We moved house in", "The results arrive in", "Snow usually falls in",
    "The harvest begins in", "Their wedding is planned for", "The fiscal year ends in",
    "The museum reopens in", "He was born in", "The exhibition closes in",
    "The trial starts in", "Rent is due at the start of", "The conference is held every",
    "They planted the garden in", "The lease expires in", "The storm hit in",
    "Tickets go on sale in", "The bridge opened in", "The marathon is run each",
    "She graduated in", "The report was published in", "Our vacation is booked for",
    "The orchard blooms in", "The election is scheduled for", "The shop holds its sale in",
    "The ice melts in", "The album was released in", "We adopted the puppy in",
    "The course begins in", "The river floods in", "The team was founded in",
    "His contract renews in", "The comet is visible in", "The treaty was signed in",
    "The camp opens in", "The tax deadline is in", "The migration starts in",
    "The film premieres in", "The audit takes place in", "Construction resumes in",
    "The garden party is held in", "The first frost comes in", "The concert tour ends in",
    "The new policy takes effect in", "Classes finish in", "The bakery closes for repairs in",
    "The annual meeting is in", "The swimming pool opens in",
)


def panel_heads(feature: str) -> tuple[str, ...]:
    """The committed heads: E1's weekday fit and base heads, or the month heads above."""
    if feature == "weekday":
        return E1.WEEKDAY_FIT_HEADS + E1.WEEKDAY_BASE_HEADS
    if feature == "month":
        return MONTH_HEADS
    raise ValueError(f"no committed panel heads for feature {feature!r}")


def panel_prompts(feature: str) -> list[tuple[int, int, str, str]]:
    """``(head index, label index, template, label)`` for every head and label."""
    labels = CALENDAR.task_from_name(feature).labels
    return [
        (head_index, label_index, f"{head} {{label}}", label)
        for head_index, head in enumerate(panel_heads(feature))
        for label_index, label in enumerate(labels)
    ]


def prompt_bank_sha256(features: tuple[str, ...]) -> str:
    payload = [
        {
            "feature": feature,
            "heads": list(panel_heads(feature)),
            "labels": list(CALENDAR.task_from_name(feature).labels),
        }
        for feature in features
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def panel_chart(x: np.ndarray, dim: int) -> tuple[np.ndarray, float]:
    """The ``dim``-dimensional PCA chart of ``x`` and the variance it keeps."""
    mean, lift = MATRIX.chart_projection(x, dim)
    chart = MATRIX.to_chart(x, mean, lift)
    total = float(((x - mean) ** 2).sum())
    if not total > 0.0:
        raise ValueError("a panel cloud with no variance has no chart")
    return chart, float((chart**2).sum()) / total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="local Hugging Face snapshot of the model")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    protocol = PANEL_PROTOCOL
    args.out.mkdir(parents=True, exist_ok=True)
    revision, model_sha256 = LEDGER.snapshot_identity(args.snapshot)
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot)
    model = AutoModelForCausalLM.from_pretrained(args.snapshot, dtype=torch.float32).to("cuda:0")
    model.eval()
    model.requires_grad_(False)
    layer = CALENDAR.resolve_layers(model)[protocol["layer"]]
    hook_module = next(name for name, module in model.named_modules() if module is layer)
    log(f"model {protocol['model']} revision {revision}; hook module {hook_module}")

    charts = {}
    for feature in protocol["features"]:
        prompts = panel_prompts(feature)
        rows = []
        for _head_index, _label_index, template, label in prompts:
            ids, position = MATRIX.label_position(tokenizer, template, label)
            activation, _logits = E1.run_clean_at(model, layer, ids, position)
            rows.append(activation.to(torch.float64).numpy())
        x = np.stack(rows)
        tag = f"{feature}_L{protocol['layer']}"
        np.savez(
            args.out / f"{tag}_raw.npz",
            X=x,
            head_index=np.asarray([prompt[0] for prompt in prompts]),
            label_index=np.asarray([prompt[1] for prompt in prompts]),
        )
        chart, kept = panel_chart(x, protocol["chart_dim"])
        np.save(args.out / f"{tag}.npy", chart)
        charts[feature] = {"rows": int(x.shape[0]), "raw_dim": int(x.shape[1]), "explained_variance": kept}
        log(f"{tag}: {x.shape[0]} rows of width {x.shape[1]}; {protocol['chart_dim']}-dim chart keeps {kept:.4f}")

    sidecar = {
        **protocol,
        "features": list(protocol["features"]),
        "model_revision": revision,
        "model_sha256": model_sha256,
        "hook_module": hook_module,
        "prompt_bank_sha256": prompt_bank_sha256(protocol["features"]),
        "driver_sha256": LEDGER.file_sha256(Path(__file__)),
        "charts": charts,
    }
    (args.out / "panel_acts.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    log(f"PANEL_ACTS {json.dumps(charts, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
