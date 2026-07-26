#!/usr/bin/env python3
"""#2263 / #2234 — steering dosimetry and on-manifold steering on REAL activations.

Both issues are blocked on the same thing: gam's manifold-SAE fit does not
converge on a real activation cloud, so `structure_certificate` and the dose
machinery downstream of it are never reached.  But the two questions those
issues actually ask are not questions about the fit:

  #2263  Is the shipped dose law  predicted_nats = 1/2 * delta^T M delta  the
         right law on a real LLM, and over what radius?  The law is a property
         of the MODEL's readout, not of how `delta` was chosen.
  #2234  Does moving a real activation ALONG a chart cost less collateral damage
         than moving it the same ambient distance off the chart?  That is a
         question about the cloud's geometry, not about who fitted the chart.

So both are measured here with the intervention directions supplied by
fit-free geometry (train-only PCA of the harvest, and a circle chart fitted to a
real token cloud by its own top-2 plane), on Qwen3.5-4B-Base at L16, on the
DECLARED #2502 document split.  The readout KL is EXACT: two forward passes of
the real model, softmax to softmax, no low-rank surrogate anywhere.

The quadratic coefficient  c = lim_{a->0} 2*KL(a)/a^2 = d^T F_h d  is the exact
object the shipped predictor estimates with a rank-r pullback, and it is
recovered here without any Fisher machinery at all — as the small-dose limit of
the measured curve.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
class Patcher:
    """Runs Qwen with an additive intervention on the layer-L block output.

    The harvest hooks the same object (`layers[L]` forward output, i.e. the
    residual stream after block L), so an intervention here is expressed in the
    exact coordinates every harvested row is written in.
    """

    def __init__(self, model_name, layer, device="cuda:0", dtype="bf16"):
        import torch
        import transformers

        self.torch = torch
        dt = {"bf16": torch.bfloat16, "fp16": torch.float16,
              "fp32": torch.float32}[dtype]
        model = None
        errs = []
        for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText",
                     "AutoModel"):
            try:
                model = getattr(transformers, name).from_pretrained(
                    model_name, torch_dtype=dt, trust_remote_code=True,
                    device_map=device)
                log(f"loaded via {name}")
                break
            except Exception as exc:  # noqa: BLE001
                errs.append(f"{name}: {type(exc).__name__}: {exc}")
        if model is None:
            raise SystemExit("could not load model:\n" + "\n".join(errs))
        model.eval()
        self.model = model
        best = None
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.ModuleList) and (
                    best is None or len(mod) > len(best[1])):
                best = (name, mod)
        self.layers = best[1]
        self.n_layers = len(self.layers)
        if not 0 <= layer < self.n_layers:
            raise SystemExit(f"layer {layer} out of range {self.n_layers}")
        self.layer = layer
        self._delta = None       # (B, D) added at self._pos, or None
        self._pos = None
        self._captured = None

        def hook(_m, _i, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._captured = hidden.detach()
            if self._delta is None:
                return output
            hidden = hidden.clone()
            hidden[:, self._pos, :] += self._delta.to(hidden.dtype)
            if isinstance(output, tuple):
                return (hidden,) + tuple(output[1:])
            return hidden

        self.handle = self.layers[layer].register_forward_hook(hook)

    def logits(self, ids, *, pos=None, delta=None, want_all_positions=False):
        """ids (B,T) long tensor; delta (B,D) or None. Returns log-softmax."""
        torch = self.torch
        self._delta = delta
        self._pos = pos
        with torch.inference_mode():
            out = self.model(input_ids=ids, use_cache=False)
        self._delta = None
        # Slice BEFORE widening: out.logits is (B, T, V) and V is ~151k, so a
        # blanket .float() on a 512-token context is gigabytes for nothing.
        if want_all_positions:
            return torch.log_softmax(out.logits.float(), dim=-1)
        return torch.log_softmax(out.logits[:, pos, :].float(), dim=-1)

    def capture(self, ids, pos):
        """Unpatched residual-stream row at (batch 0, pos) after block L."""
        self.logits(ids, pos=pos, delta=None)
        return self._captured[:, pos, :].float()


def window_ids(token_ids, s, pos, seq_len, before, after, device):
    """Context window [pos-before, pos+after] of sequence `s`, as (1, T).

    The harvest packs documents into fixed 512-token sequences, so a row is
    (sequence, position) and its context is recoverable exactly.  Truncating to
    a window keeps the (B, T, V) logit tensor affordable, and the base
    activation is recomputed inside the same window so nothing is ever compared
    across contexts.

    `after` matters for collateral damage: the model is causal, so an edit at
    position `pos` cannot reach any earlier position — its collateral lives
    strictly in the positions that follow, and a window ending at `pos` would
    report a collateral of exactly zero by construction.
    """
    import torch

    lo = max(0, pos - before)
    hi = min(seq_len - 1, pos + after)
    seg = token_ids[s * seq_len + lo:s * seq_len + hi + 1].astype(np.int64)
    return torch.tensor(seg, device=device).unsqueeze(0), int(pos - lo)


def kl_rows(logp_base, logp_new):
    """KL(base || new) per row, exact, from log-softmax rows."""
    p = logp_base.exp()
    return (p * (logp_base - logp_new)).sum(-1)


# --------------------------------------------------------------------------- #
def load_split(harvest, split_module):
    import sys
    sys.path.insert(0, split_module)
    from issue_2502_doc_split import row_side, split_manifest

    doc_ids = np.load(Path(harvest) / "doc_ids.npy")
    man = split_manifest(doc_ids)
    return row_side(doc_ids), man


def train_pca(acts, side, *, n_rows, n_comp, seed=0):
    """Top-`n_comp` principal directions of the TRAIN side (no test leakage)."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(np.flatnonzero(~side), n_rows, replace=False))
    x = np.asarray(acts[idx], dtype=np.float32)
    mu = x.mean(0)
    xc = x - mu
    cov = (xc.T @ xc) / max(xc.shape[0] - 1, 1)
    w, v = np.linalg.eigh(cov.astype(np.float64))
    order = np.argsort(w)[::-1][:n_comp]
    return mu, v[:, order].T.astype(np.float32), w[order]


# --------------------------------------------------------------------------- #
def mode_dose(args, patcher, acts, side, token_ids, man, emit):
    """#2263: exact dose-response law and the readout-KL validity radius."""
    import torch

    seq_len = args.seq_len
    mu, dirs_pca, eig = train_pca(acts, side, n_rows=args.pca_rows,
                                  n_comp=args.n_dirs, seed=0)
    rng = np.random.default_rng(11)
    dirs_rand = rng.standard_normal((args.n_dirs, acts.shape[1])).astype(np.float32)
    dirs_rand /= np.linalg.norm(dirs_rand, axis=1, keepdims=True)

    families = {"pca": dirs_pca, "random": dirs_rand}

    # Panel rows: held-out, and not at a sequence boundary position.
    test_rows = np.flatnonzero(side)
    prng = np.random.default_rng(3)
    cand = prng.choice(test_rows, size=args.n_rows * 8, replace=False)
    cand = [int(r) for r in cand if (r % seq_len) >= args.min_pos][:args.n_rows]
    log(f"dose panel: {len(cand)} held-out rows")

    rhos = [float(r) for r in args.rhos.split(",")]

    for row in cand:
        s, pos0 = row // seq_len, row % seq_len
        ids, pos = window_ids(token_ids, s, pos0, seq_len, args.window_before,
                              0, args.device)
        x0 = patcher.capture(ids, pos)[0]          # (D,) float32 cuda
        x0n = float(x0.norm())

        for fam, dirs in families.items():
            for gi in range(dirs.shape[0]):
                u = torch.from_numpy(dirs[gi]).to(args.device)
                u = u / u.norm()
                amps = [r * x0n for r in rhos]
                # Two ZERO-dose rows ride in the same batch as the doses. The
                # base distribution is then computed by the same kernel launch
                # at the same batch shape as every perturbed row (a base taken
                # from a separate B=1 forward differs by kernel choice alone),
                # and KL(row0 || row1) is this record's own NOISE FLOOR: the
                # readout KL between two bit-identical interventions. Any dose
                # whose KL is not clear of that floor is measuring arithmetic.
                zero = torch.zeros_like(u)
                deltas = torch.stack([zero, zero] + [a * u for a in amps])
                B = deltas.shape[0]
                ids_b = ids.expand(B, -1).contiguous()
                logp = patcher.logits(ids_b, pos=pos, delta=deltas)
                base = logp[0]
                floor = float(kl_rows(base.unsqueeze(0),
                                      logp[1].unsqueeze(0))[0])
                kls = kl_rows(base.unsqueeze(0).expand(B - 2, -1),
                              logp[2:]).double().cpu().numpy()

                # The discriminator for "is the quadratic law holding here" is
                # the LOCAL log-log slope d log KL / d log a, which is exactly 2
                # wherever KL = 1/2 c a^2 and nothing else contributes. Below
                # the window the perturbation is being lost to the residual
                # stream's own arithmetic (slope -> 0); above it the readout
                # saturates (slope < 2). Emitting the slopes rather than a
                # single anchored coefficient keeps the raw ladder the datum,
                # so the window can be re-read without re-running the model.
                slopes = []
                for j in range(1, len(amps)):
                    if kls[j] > 0 and kls[j - 1] > 0:
                        slopes.append(float(
                            (math.log(kls[j]) - math.log(kls[j - 1]))
                            / (math.log(amps[j]) - math.log(amps[j - 1]))))
                    else:
                        slopes.append(float("nan"))
                # quadratic window: the longest run of consecutive intervals
                # whose slope is within `slope_tol` of 2
                best = (0, None)
                run_start = None
                for j, sl in enumerate(slopes + [float("nan")]):
                    ok = math.isfinite(sl) and abs(sl - 2.0) <= args.slope_tol
                    if ok and run_start is None:
                        run_start = j
                    elif not ok and run_start is not None:
                        if j - run_start > best[0]:
                            best = (j - run_start, (run_start, j))
                        run_start = None
                c_hat, win = None, None
                if best[1] is not None:
                    lo, hi = best[1]           # intervals [lo, hi) -> points lo..hi
                    win = [rhos[lo], rhos[hi]]
                    cs = [2.0 * float(kls[j]) / (amps[j] ** 2)
                          for j in range(lo, hi + 1)]
                    c_hat = float(np.exp(np.mean(np.log(cs))))
                emit({"record": "dose", "row": int(row), "seq": int(s),
                      "pos": int(pos), "family": fam, "dir": int(gi),
                      "x_norm": x0n, "identical_pair_kl": floor,
                      "rhos": rhos, "kl": [float(k) for k in kls],
                      "loglog_slopes": slopes,
                      "quadratic_window_rho": win, "c_hat": c_hat})
        log(f"row {row} done")


# --------------------------------------------------------------------------- #
def fit_circle_chart(cloud):
    """Fit-free circle chart: top-2 plane of a token cloud, angle per row.

    Returns (center, basis (2,D), angles, radii).  This is the geometry a
    1-harmonic circle atom parameterises, obtained without the walled solver.
    """
    mu = cloud.mean(0)
    xc = cloud - mu
    cov = (xc.T @ xc) / max(xc.shape[0] - 1, 1)
    w, v = np.linalg.eigh(cov.astype(np.float64))
    order = np.argsort(w)[::-1][:2]
    B = v[:, order].T.astype(np.float32)          # (2, D)
    coords = xc @ B.T                             # (n, 2)
    ang = np.arctan2(coords[:, 1], coords[:, 0])
    rad = np.linalg.norm(coords, axis=1)
    return mu, B, ang, rad, w[order]


def mode_chart(args, patcher, acts, side, token_ids, man, emit):
    """#2234: dose in radians along a real chart vs an off-chart move."""
    import torch

    seq_len = args.seq_len
    tok = patcher_tokenizer(args)
    class_ids = class_token_ids(tok, args.cyclic_class)
    log(f"class token ids: {class_ids}")

    in_class = np.isin(token_ids, np.asarray(class_ids, dtype=token_ids.dtype))
    cloud_rows_train = np.flatnonzero(in_class & ~side)
    cloud_rows_test = np.flatnonzero(in_class & side)
    log(f"class cloud: {cloud_rows_train.size} train rows, "
        f"{cloud_rows_test.size} held-out rows")
    if cloud_rows_train.size < 64 or cloud_rows_test.size < 8:
        raise SystemExit("class cloud too small")

    cloud = np.asarray(acts[cloud_rows_train], dtype=np.float32)
    mu, Bc, ang, rad, eig = fit_circle_chart(cloud)
    r_med = float(np.median(rad))
    log(f"chart: r_med={r_med:.3f} plane eigenvalues={eig.tolist()} "
        f"angle spread={float(ang.std()):.3f} rad")
    emit({"record": "chart_fit", "cyclic_class": args.cyclic_class,
          "n_cloud": int(cloud.shape[0]),
          "r_median": r_med, "plane_eigenvalues": [float(e) for e in eig],
          "class_token_ids": [int(t) for t in class_ids]})

    prng = np.random.default_rng(5)
    panel = [int(r) for r in prng.choice(cloud_rows_test,
                                         min(args.n_rows, cloud_rows_test.size),
                                         replace=False)
             if (r % seq_len) >= args.min_pos]
    log(f"chart panel: {len(panel)} held-out class rows")

    dthetas = [float(d) for d in args.dthetas.split(",")]
    Bt = torch.from_numpy(Bc).to(args.device)             # (2, D)
    mu_t = torch.from_numpy(mu).to(args.device)
    class_t = torch.tensor(class_ids, device=args.device, dtype=torch.long)

    for row in panel:
        s, pos0 = row // seq_len, row % seq_len
        ids, pos = window_ids(token_ids, s, pos0, seq_len, args.window_before,
                              args.window_after, args.device)
        x0 = patcher.capture(ids, pos)[0]
        logp_all_base = patcher.logits(ids, pos=pos, delta=None,
                                       want_all_positions=True)[0]
        logp_base = logp_all_base[pos]

        c = (x0 - mu_t) @ Bt.T                            # (2,) chart coords
        r0 = float(c.norm())
        th0 = float(torch.atan2(c[1], c[0]))
        # component of the row off the chart plane stays untouched
        for dth in dthetas:
            th1 = th0 + dth
            c1 = torch.tensor([r0 * math.cos(th1), r0 * math.sin(th1)],
                              device=args.device)
            d_on = (c1 - c) @ Bt                          # (D,) on-chart move
            norm = float(d_on.norm())
            # matched-norm off-chart control: random direction orthogonal to
            # the chart plane, same ambient length, so the ONLY difference is
            # whether the move stays on the fitted circle.
            g = torch.randn(x0.shape[0], device=args.device,
                            generator=torch.Generator(device=args.device)
                            .manual_seed(1000 + int(row) + int(dth * 1e4)))
            g = g - (g @ Bt.T) @ Bt
            d_off = g / g.norm() * norm
            # matched-norm RADIAL control: move along the chart plane but off
            # the circle (changes r, not theta) — isolates "on the plane" from
            # "along the curve".
            d_rad = (c / c.norm()) @ Bt * norm

            deltas = torch.stack([d_on, d_off, d_rad])
            ids_b = ids.expand(3, -1).contiguous()
            logp_all = patcher.logits(ids_b, pos=pos, delta=deltas,
                                      want_all_positions=True)
            arms = {}
            for j, nm in enumerate(("on_chart", "off_chart", "radial")):
                lp = logp_all[j]
                kl_target = float(kl_rows(logp_base.unsqueeze(0),
                                          lp[pos].unsqueeze(0))[0])
                # collateral: KL at every OTHER position of the same sequence,
                # which the edit reaches only through attention.
                mask = torch.zeros(lp.shape[0], dtype=torch.bool,
                                   device=lp.device)
                mask[pos + 1:] = True
                kl_other = kl_rows(logp_all_base[mask], lp[mask])
                # specificity: how much of the target-position logit motion
                # landed on the class the chart was built from.
                dlog = lp[pos] - logp_base
                arms[nm] = {
                    "kl_target": kl_target,
                    "kl_other_mean": float(kl_other.mean()),
                    "kl_other_max": float(kl_other.max()),
                    "kl_other_sum": float(kl_other.sum()),
                    "class_logprob_shift": float(dlog[class_t].mean()),
                    "all_logprob_shift_abs": float(dlog.abs().mean()),
                }
            emit({"record": "chart_dose", "row": int(row), "seq": int(s),
                  "pos": int(pos), "dtheta": dth, "ambient_norm": norm,
                  "r_chart": r0, "theta0": th0, "arms": arms})
        log(f"chart row {row} done")


def patcher_tokenizer(args):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)


CYCLIC_CLASSES = {
    # Cyclic token classes: the chart being fitted is a CIRCLE, so the class
    # has to have a cyclic order for "dose in radians" to mean anything.
    "month": [" January", " February", " March", " April", " May", " June",
              " July", " August", " September", " October", " November",
              " December"],
    "weekday": [" Monday", " Tuesday", " Wednesday", " Thursday", " Friday",
                " Saturday", " Sunday"],
}


def class_token_ids(tok, class_name):
    names = CYCLIC_CLASSES[class_name]
    out = []
    for n in names:
        ids = tok.encode(n, add_special_tokens=False)
        if len(ids) == 1:
            out.append(ids[0])
    if len(out) < 5:
        raise SystemExit(f"{class_name} names are not single tokens: {out}")
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dose", "chart"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--harvest", default=os.path.expanduser("~/i2502/harvest"))
    ap.add_argument("--split-module",
                    default=os.path.expanduser("~/i2502-baselines"))
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="fp16",
                    help="bf16 has an 8-bit mantissa: a small intervention is "
                         "lost to the residual stream's own rounding before it "
                         "reaches the readout, and the dose ladder measures "
                         "arithmetic instead of the model. fp16 lowers that "
                         "floor by ~30x and opens the quadratic window.")
    ap.add_argument("--n-rows", type=int, default=12)
    ap.add_argument("--n-dirs", type=int, default=4)
    ap.add_argument("--pca-rows", type=int, default=40000)
    ap.add_argument("--min-pos", type=int, default=16)
    ap.add_argument("--cyclic-class", default="month",
                    choices=sorted(CYCLIC_CLASSES))
    ap.add_argument("--window-before", type=int, default=128)
    ap.add_argument("--window-after", type=int, default=48,
                    help="collateral positions after the edit (causality: an "
                         "edit reaches only positions >= its own)")
    ap.add_argument("--rhos",
                    default="0.002,0.005,0.01,0.02,0.035,0.06,0.1,0.18,0.3,"
                            "0.5,0.8,1.2")
    ap.add_argument("--slope-tol", type=float, default=0.15,
                    help="how far the local log-log slope may sit from 2 and "
                         "still count as inside the quadratic window")
    ap.add_argument("--dthetas", default="0.05,0.1,0.2,0.4,0.8,1.6")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    import torch

    side, man = load_split(args.harvest, args.split_module)
    acts = np.load(Path(args.harvest) / f"resid_L{args.layer}.npy",
                   mmap_mode="r")
    token_ids = np.load(Path(args.harvest) / "token_ids.npy")
    prov = {
        "node": platform.node(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "cuda_device": torch.cuda.get_device_name(0),
        "model": args.model, "layer": args.layer, "dtype": args.dtype,
        "split_hash": man["split_hash"],
        "harvest_doc_digest": man["harvest_doc_digest"],
    }
    log(f"provenance {json.dumps(prov)}")

    def emit(rec):
        rec["provenance"] = prov
        with open(outp, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

    patcher = Patcher(args.model, args.layer, device=args.device,
                      dtype=args.dtype)
    log(f"model has {patcher.n_layers} decoder layers; patching block "
        f"{args.layer} output")

    if args.mode == "dose":
        mode_dose(args, patcher, acts, side, token_ids, man, emit)
    else:
        mode_chart(args, patcher, acts, side, token_ids, man, emit)
    log(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
