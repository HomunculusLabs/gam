"""#2502 GPU stage A: capture everything the (un-reloadable) flagship fit
process will need, so all LLM work is decoupled from the Rust model lifetime.

Saves ~/i2502/stage_a.npz + stage_a_meta.json:
  - weekday + month calendar clouds (fit + base): last-token L16 activations,
    labels, and full base logits (fp16)
  - fresh wikitext-validation token batch (n_seqs x 512) + its L16 hidden
    states (fp16) + base next-token CE
"""
import json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "Qwen/Qwen3.5-4B-Base"
LAYER = 16
N_SEQS, SEQ_LEN = 40, 512
VAL_BATCH = 2
WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MON = ["January", "February", "March", "April", "May", "June", "July",
       "August", "September", "October", "November", "December"]
WEEK_FIT = (
    "Today is {label}. Tomorrow is", "If today is {label}, then tomorrow is",
    "The weekday after {label} is", "On a weekly calendar, {label} is followed by",
    "Yesterday was {label}, so today is", "The day that comes right after {label} is",
    "After {label} comes", "Counting forward from {label}, the next day is",
    "A day later than {label} is", "Following {label} on the calendar is")
WEEK_BASE = ("Starting on {label}, the next day is",
             "Calendar note: the day after {label} is")
MONTH_FIT = (
    "This month is {label}. Next month is", "The month after {label} is",
    "In the calendar year, {label} is followed by", "After {label} comes",
    "One month later than {label} is", "The month right after {label} is",
    "Following {label} in the year is",
    "Counting forward from {label}, the next month is")
MONTH_BASE = ("Starting in {label}, the next month is",
              "Calendar note: the month after {label} is")

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda:0")
model.eval()
best = None
for _n, mod in model.named_modules():
    if isinstance(mod, torch.nn.ModuleList) and (best is None or len(mod) > len(best)):
        best = mod
layer = best[LAYER]

cap = {}


def hook(_m, _i, output):
    h = output[0] if isinstance(output, tuple) else output
    cap["h"] = h.detach()


handle = layer.register_forward_hook(hook)


def run(prompt):
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
    with torch.inference_mode():
        out = model(**enc, use_cache=False)
    pos = enc["input_ids"].shape[1] - 1
    return (cap["h"][0, pos].float().cpu().numpy(),
            out.logits[0, pos].float().cpu().numpy())


save = {}
meta = {"model": MODEL, "layer": LAYER}
for cyc, labels, fit_t, base_t in (("week", WEEK, WEEK_FIT, WEEK_BASE),
                                   ("month", MON, MONTH_FIT, MONTH_BASE)):
    for part, templates in (("fit", fit_t), ("base", base_t)):
        acts, labs, logits, prompts = [], [], [], []
        for tmpl in templates:
            for i, lab in enumerate(labels):
                a, lg = run(tmpl.format(label=lab))
                acts.append(a)
                labs.append(i)
                logits.append(lg)
                prompts.append(tmpl.format(label=lab))
        save[f"{cyc}_{part}_X"] = np.stack(acts)
        save[f"{cyc}_{part}_lab"] = np.array(labs, dtype=np.int32)
        if part == "base":
            save[f"{cyc}_base_logits"] = np.stack(logits).astype(np.float16)
        meta[f"{cyc}_{part}_prompts"] = prompts
    meta[f"{cyc}_cand_ids"] = [tok.encode(" " + w, add_special_tokens=False)[0]
                               for w in labels]
    print(f"captured {cyc}", flush=True)

# fresh validation batch
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                  split="validation", streaming=True)
buf, seqs = [], []
for ex in ds:
    if not ex["text"].strip():
        continue
    buf.extend(tok.encode(ex["text"], add_special_tokens=False))
    while len(buf) >= SEQ_LEN and len(seqs) < N_SEQS:
        seqs.append(buf[:SEQ_LEN])
        buf = buf[SEQ_LEN:]
    if len(seqs) >= N_SEQS:
        break
ids = torch.tensor(seqs, dtype=torch.long, device="cuda:0")
hs, ces = [], []
import torch.nn.functional as F
with torch.inference_mode():
    for i in range(0, len(ids), VAL_BATCH):
        out = model(input_ids=ids[i:i + VAL_BATCH], use_cache=False)
        hs.append(cap["h"].float().cpu())
        ces.append(F.cross_entropy(
            out.logits[:, :-1].float().reshape(-1, out.logits.shape[-1]),
            ids[i:i + VAL_BATCH, 1:].reshape(-1), reduction="sum").item())
        del out
handle.remove()
save["val_ids"] = ids.cpu().numpy().astype(np.int32)
save["val_h"] = torch.cat(hs).numpy().astype(np.float16)
meta["val_base_ce"] = sum(ces) / (ids.shape[0] * (SEQ_LEN - 1))
print("val base CE", meta["val_base_ce"], flush=True)

np.savez_compressed(os.path.expanduser("~/i2502/stage_a.npz"), **save)
json.dump(meta, open(os.path.expanduser("~/i2502/stage_a_meta.json"), "w"))
print("STAGE_A DONE", flush=True)
