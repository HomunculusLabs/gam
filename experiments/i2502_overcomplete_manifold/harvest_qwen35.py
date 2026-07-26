"""Harvest Qwen3.5-4B-Base residual-stream activations on wikitext for #2502.

Saves, per requested layer L: resid_L{L}.npy (N, D) fp16, plus shared
token_ids.npy, doc_ids.npy, pos_in_seq.npy and docs.jsonl (id -> text) so any
row can be mapped back to its token-in-context for interpretation.
"""
import argparse, json, os, time
import numpy as np


def resolve_decoder_layers(model):
    import torch
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and (best is None or len(mod) > len(best[1])):
            best = (name, mod)
    if best is None:
        raise ValueError("no ModuleList found")
    print(f"[harvest] decoder layers at {best[0]!r} (n={len(best[1])})", flush=True)
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 16, 22])
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n-tokens", type=int, default=600_000)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--out", default=os.path.expanduser("~/i2502/harvest"))
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = None
    errs = []
    for loader_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"):
        try:
            import transformers
            loader = getattr(transformers, loader_name)
            model = loader.from_pretrained(
                args.model, torch_dtype=dtype, trust_remote_code=True, device_map="cuda:0"
            )
            print(f"[harvest] loaded via {loader_name}", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            errs.append(f"{loader_name}: {type(e).__name__}: {e}")
    if model is None:
        raise SystemExit("could not load model:\n" + "\n".join(errs))
    model.eval()
    layers = resolve_decoder_layers(model)
    n_layers = len(layers)
    for L in args.layers:
        assert 0 <= L < n_layers, f"layer {L} out of range {n_layers}"

    # hooks capture the BLOCK OUTPUT hidden state (residual stream after layer L)
    captured = {}

    def make_hook(L):
        def hook(_m, _i, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[L] = hidden.detach()
        return hook

    handles = [layers[L].register_forward_hook(make_hook(L)) for L in args.layers]

    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train",
                      streaming=True)

    # pack documents into fixed-length sequences, tracking doc boundaries
    seqs, doc_texts = [], []
    buf_ids, buf_doc, buf_pos = [], [], []
    cur_doc_id = -1
    total = 0
    for ex in ds:
        text = ex["text"]
        if not text.strip():
            continue
        cur_doc_id += 1
        doc_texts.append(text)
        ids = tok.encode(text, add_special_tokens=False)
        for j, t in enumerate(ids):
            buf_ids.append(t)
            buf_doc.append(cur_doc_id)
            buf_pos.append(j)
            if len(buf_ids) == args.seq_len:
                seqs.append((buf_ids, buf_doc, buf_pos))
                buf_ids, buf_doc, buf_pos = [], [], []
                total += args.seq_len
        if total >= args.n_tokens:
            break
    print(f"[harvest] packed {len(seqs)} seqs x {args.seq_len} = {total} tokens "
          f"from {cur_doc_id + 1} docs", flush=True)

    with open(os.path.join(args.out, "docs.jsonl"), "w") as f:
        for i, t in enumerate(doc_texts):
            f.write(json.dumps({"doc_id": i, "text": t}) + "\n")

    N = len(seqs) * args.seq_len
    D = model.config.hidden_size if hasattr(model.config, "hidden_size") else \
        model.config.text_config.hidden_size
    acts = {L: np.empty((N, D), dtype=np.float16) for L in args.layers}
    all_ids = np.empty(N, dtype=np.int32)
    all_doc = np.empty(N, dtype=np.int32)
    all_pos = np.empty(N, dtype=np.int32)

    t0 = time.time()
    row = 0
    with torch.inference_mode():
        for b0 in range(0, len(seqs), args.batch):
            chunk = seqs[b0:b0 + args.batch]
            ids = torch.tensor([c[0] for c in chunk], dtype=torch.long, device="cuda:0")
            model(input_ids=ids, use_cache=False)
            nrows = ids.numel()
            for L in args.layers:
                h = captured.pop(L)  # (B, T, D)
                acts[L][row:row + nrows] = (
                    h.reshape(-1, h.shape[-1]).to(torch.float16).cpu().numpy()
                )
            all_ids[row:row + nrows] = np.concatenate([c[0] for c in chunk])
            all_doc[row:row + nrows] = np.concatenate([c[1] for c in chunk])
            all_pos[row:row + nrows] = np.concatenate([c[2] for c in chunk])
            row += nrows
            if (b0 // args.batch) % 10 == 0:
                el = time.time() - t0
                print(f"[harvest] {row}/{N} rows  {row/max(el,1e-9):.0f} tok/s", flush=True)

    for h in handles:
        h.remove()
    for L in args.layers:
        np.save(os.path.join(args.out, f"resid_L{L}.npy"), acts[L][:row])
    np.save(os.path.join(args.out, "token_ids.npy"), all_ids[:row])
    np.save(os.path.join(args.out, "doc_ids.npy"), all_doc[:row])
    np.save(os.path.join(args.out, "pos_in_seq.npy"), all_pos[:row])
    meta = dict(model=args.model, layers=args.layers, seq_len=args.seq_len,
                n_rows=int(row), hidden=int(D), n_layers=int(n_layers),
                corpus="wikitext-103-raw-v1/train", dtype_store="fp16",
                wall_s=round(time.time() - t0, 1))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[harvest] DONE", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
