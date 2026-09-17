"""Export one transformer MLP block and its executed outputs (issue #2946).

Thin PyTorch wrapper. It loads a cached checkpoint, runs forward passes and
writes arrays; nothing here does modelling math. The declared law (baseline
``h0``, factor ``L``), the sampled latents, the analytic ``V(P)``/``E(P)`` and
the Monte Carlo estimators all live in Rust, which reads these files.

Two stages, each one job:

``harvest``  runs the model's first ``layer + 1`` blocks over a named context
             set and writes, into ``--out-dir``:
               post_norm.npy       float32 (rows x hidden) MLP input at ``layer``,
                                   i.e. the post-norm activation the MLP sees
               <param>.npy         float64, every parameter of that MLP under
                                   its torch name (e.g. ``gate_proj.weight``)
               meta.json           checkpoint revision, config fields, context
                                   set definition, row count, library versions

``execute``  runs that MLP alone, in float64, on rows ``h`` that Rust wrote
             (``h = h0 + L z``) and writes ``F(h)`` as float64 (rows x hidden).

Checkpoints load with ``local_files_only=True``: the exporter never downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer


def _set_threads() -> None:
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpus is not None:
        torch.set_num_threads(int(cpus))


def _truncated_model(model_id: str, revision: str, layer: int, dtype: torch.dtype):
    """Load the decoder with only blocks ``0..=layer`` and no vocabulary head.

    Nothing past the exported MLP is needed, so the resident weights are the
    embedding plus ``layer + 1`` blocks.
    """
    cfg = AutoConfig.from_pretrained(model_id, revision=revision, local_files_only=True)
    if not 0 <= layer < cfg.num_hidden_layers:
        raise SystemExit(f"--layer {layer} outside 0..{cfg.num_hidden_layers - 1}")
    full_depth = cfg.num_hidden_layers
    cfg.num_hidden_layers = layer + 1
    model = AutoModel.from_pretrained(
        model_id, revision=revision, config=cfg, dtype=dtype, local_files_only=True
    )
    model.eval()
    return cfg, full_depth, model


def _md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def harvest(args: argparse.Namespace) -> int:
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    cfg, full_depth, model = _truncated_model(args.model, args.revision, args.layer, torch.float32)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=True)
    backbone = model
    mlp = backbone.layers[args.layer].mlp
    hidden = int(cfg.hidden_size)
    print(
        f"[harvest] model={args.model} type={cfg.model_type} act={cfg.hidden_act} "
        f"hidden={hidden} intermediate={cfg.intermediate_size} layer={args.layer}/{full_depth}",
        flush=True,
    )

    # The context set: the named split's records in order, each tokenized without
    # special tokens and concatenated, cut into consecutive windows.
    ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
    need = args.windows * args.seq_len
    stream: list[int] = []
    records_used = 0
    for record in ds:
        if len(stream) >= need:
            break
        stream.extend(tok(record["text"], add_special_tokens=False)["input_ids"])
        records_used += 1
    if len(stream) < need:
        raise SystemExit(f"split holds {len(stream)} tokens, the context set needs {need}")
    windows = torch.tensor(stream[:need], dtype=torch.long).view(args.windows, args.seq_len)

    keep = args.seq_len - args.skip_positions
    rows = args.windows * keep
    acts = np.lib.format.open_memmap(
        os.path.join(args.out_dir, "post_norm.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(rows, hidden),
    )
    captured: dict[str, torch.Tensor] = {}

    def grab(_module, inputs):
        captured["h"] = inputs[0]

    handle = mlp.register_forward_pre_hook(grab)
    filled = 0
    with torch.no_grad():
        for start in range(0, args.windows, args.batch):
            backbone(input_ids=windows[start : start + args.batch], use_cache=False)
            block = captured["h"][:, args.skip_positions :, :].reshape(-1, hidden)
            acts[filled : filled + block.shape[0]] = block.numpy()
            filled += block.shape[0]
            print(f"[harvest] windows {start + block.shape[0] // keep}/{args.windows}", flush=True)
    handle.remove()
    acts.flush()

    params = {}
    for name, p in mlp.named_parameters():
        path = os.path.join(args.out_dir, f"{name}.npy")
        np.save(path, p.detach().to(torch.float64).numpy())
        params[name] = list(p.shape)

    meta = {
        "stage": "harvest",
        "model": args.model,
        "revision": args.revision,
        "resolved_revision": getattr(cfg, "_commit_hash", None),
        "model_type": cfg.model_type,
        "hidden_act": cfg.hidden_act,
        "hidden_size": hidden,
        "intermediate_size": int(cfg.intermediate_size),
        "layer": args.layer,
        "num_hidden_layers": full_depth,
        "checkpoint_dtype": str(getattr(cfg, "torch_dtype", None)),
        "forward_dtype": "float32",
        "mlp_class": type(mlp).__name__,
        "mlp_params": params,
        "context_set": {
            "dataset": args.dataset,
            "config": args.dataset_config,
            "split": args.split,
            "records_used": records_used,
            "windows": args.windows,
            "seq_len": args.seq_len,
            "skip_positions": args.skip_positions,
        },
        "post_norm_rows": rows,
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        },
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[harvest] wrote {rows} rows and {len(params)} parameters in {meta['seconds']}s", flush=True)
    return 0


def execute(args: argparse.Namespace) -> int:
    t0 = time.time()
    cfg, full_depth, model = _truncated_model(args.model, args.revision, args.layer, torch.float32)
    mlp = model.layers[args.layer].mlp.to(torch.float64)
    del model
    inputs = np.load(args.inputs, mmap_mode="r")
    if inputs.ndim != 2 or inputs.shape[1] != cfg.hidden_size or inputs.dtype != np.float64:
        raise SystemExit(
            f"--inputs must be float64 (rows x {cfg.hidden_size}); got {inputs.dtype} {inputs.shape}"
        )
    rows = inputs.shape[0]
    outputs = np.lib.format.open_memmap(
        args.out, mode="w+", dtype=np.float64, shape=(rows, cfg.hidden_size)
    )
    with torch.no_grad():
        for start in range(0, rows, args.batch):
            h = torch.from_numpy(np.ascontiguousarray(inputs[start : start + args.batch]))
            outputs[start : start + h.shape[0]] = mlp(h).numpy()
    outputs.flush()
    meta = {
        "stage": "execute",
        "model": args.model,
        "revision": args.revision,
        "resolved_revision": getattr(cfg, "_commit_hash", None),
        "layer": args.layer,
        "mlp_class": type(mlp).__name__,
        "forward_dtype": "float64",
        "inputs": args.inputs,
        "inputs_md5": _md5(args.inputs),
        "rows": rows,
        "outputs": args.out,
        "outputs_md5": _md5(args.out),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "seconds": round(time.time() - t0, 1),
    }
    with open(args.out + ".json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[execute] {type(mlp).__name__} on {rows} rows in {meta['seconds']}s", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    h = sub.add_parser("harvest", help="post-norm activations and MLP parameters at one layer")
    h.add_argument("--model", required=True, help="checkpoint id, resolved from the local HF cache")
    h.add_argument("--revision", required=True, help="pinned checkpoint commit sha")
    h.add_argument("--layer", type=int, required=True)
    h.add_argument("--dataset", required=True)
    h.add_argument("--dataset-config", required=True)
    h.add_argument("--split", required=True)
    h.add_argument("--windows", type=int, required=True)
    h.add_argument("--seq-len", type=int, required=True)
    h.add_argument(
        "--skip-positions",
        type=int,
        required=True,
        help="leading positions of each window left out of the rows",
    )
    h.add_argument("--batch", type=int, required=True, help="windows per forward pass")
    h.add_argument("--out-dir", required=True)

    e = sub.add_parser("execute", help="run the MLP alone in float64 on rows Rust wrote")
    e.add_argument("--model", required=True)
    e.add_argument("--revision", required=True, help="pinned checkpoint commit sha")
    e.add_argument("--layer", type=int, required=True)
    e.add_argument("--inputs", required=True, help="float64 .npy (rows x hidden)")
    e.add_argument("--batch", type=int, required=True, help="rows per forward call")
    e.add_argument("--out", required=True, help="float64 .npy (rows x hidden) F(h)")

    args = ap.parse_args(argv)
    _set_threads()
    return harvest(args) if args.stage == "harvest" else execute(args)


if __name__ == "__main__":
    sys.exit(main())
