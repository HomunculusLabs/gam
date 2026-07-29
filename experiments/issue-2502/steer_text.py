"""Qualitative steering: walk an atom's fitted curve and READ the generations.

For the strongest loop and the structural sphere from the interpretable fit:
inject the decoded direction (a chord along the manifold) into layer 16 at
every generated position, sample continuations, and print them next to the
unsteered baseline and an equal-norm random control.
"""
import os, json
import numpy as np
import torch

V2 = os.path.expanduser("~/i2502v2")
d = os.path.join(V2, "b_mix1818")
man = json.load(open(d + "/manifest.json"))

lift = np.load(V2 + "/lift.npy")  # (128, D): chart -> ambient stream

from transformers import AutoModelForCausalLM, AutoTokenizer
name = "Qwen/Qwen3.5-4B-Base"
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16,
                                             trust_remote_code=True, device_map="cuda:0")
model.eval()
LAYER = 16
layers = None
for _n, mod in model.named_modules():
    if isinstance(mod, torch.nn.ModuleList) and (layers is None or len(mod) > len(layers)):
        layers = mod
block = layers[LAYER]

inject = {"vec": None}

def hook(_m, _i, output):
    if inject["vec"] is None:
        return output
    if isinstance(output, tuple):
        output[0].add_(inject["vec"])
        return output
    output.add_(inject["vec"])
    return output

handle = block.register_forward_hook(hook)

def lift_to_stream(delta128):
    return delta128 @ lift

def curve_dirs(atom):
    curve = np.fromfile(d + "/curve_%d.bin" % atom["idx"]).reshape(-1, 128)
    n = len(curve)
    # chords at three phases along the curve
    out = []
    for lo, hi, tag in [(0, n // 3, "phase 0->1/3"), (n // 3, 2 * n // 3, "1/3->2/3"),
                        (2 * n // 3, n - 1, "2/3->end")]:
        out.append((curve[hi] - curve[lo], tag))
    return out

def gen(prompt, vec, scale):
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    if vec is None:
        inject["vec"] = None
    else:
        v = torch.tensor(vec, dtype=torch.bfloat16, device="cuda:0")
        inject["vec"] = scale * v / v.norm()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=40, do_sample=True, temperature=0.8,
                             top_p=0.95, pad_token_id=tok.eos_token_id)
    inject["vec"] = None
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).replace("\n", " / ")

rng = np.random.default_rng(0)
prompts = ["The battle began when", "In 1985, the company", "The most important thing about"]
targets = [a for a in man if a["atom"] in (326, 948)] or man[:2]
SCALES = [4.0, 8.0]
for atom in targets:
    print("=" * 80)
    print("ATOM %d (%s, usage %d)" % (atom["atom"], atom["kind"], atom["usage"]))
    dirs = curve_dirs(atom)
    for prompt in prompts[:2]:
        print("-- PROMPT: %r" % prompt)
        print("   base   :", gen(prompt, None, 0.0))
        for scale in SCALES:
            for delta128, tag in dirs[:2]:
                full = lift_to_stream(delta128)
                print("   s=%d %s:" % (scale, tag), gen(prompt, full, scale))
            rand = rng.normal(size=full.shape)
            print("   s=%d random :" % scale, gen(prompt, rand, scale))
handle.remove()
print("DONE")
