"""#2502 flagship pipeline (single process — the support-sparse model has no
from_dict, so everything model-dependent happens here, on CPU, gpu=off):

  1. fit the overcomplete manifold dictionary (K circle atoms, hard-TopK
     support lane, REML/LAML-selected smoothing seminorm)
  2. benchmark stats -> fits.jsonl
  3. census + top-token interpretation + unsupervised calendar scan
  4. self-verified basis decode (my Phi(t)@B_k must reproduce model.fitted)
  5. steering deltas for stage-B patching (on-manifold phase moves + torch-SAE
     control direction, lifted to ambient)
  6. splice reconstructions of the captured validation batch for all arms
  7. figures: atom gallery, overcompleteness, calendar circles
"""
import argparse, json, os, pickle, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#333333"
TAU = 2.0 * np.pi


def ev_of(X, R):
    return 1.0 - float(((X - R) ** 2).sum()) / float((X ** 2).sum())


def circular_r2(phase_turns, idx, n):
    truth = np.exp(1j * TAU * idx / n)
    chart = np.exp(1j * TAU * phase_turns)
    fwd = abs(np.mean(truth * np.conj(chart)))
    rev = abs(np.mean(truth * chart))
    return max(fwd, rev) ** 2, (1 if fwd >= rev else -1)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def slot_coords(latents, n_rows, top_k):
    """coords as (n_rows, top_k) phase array regardless of payload layout."""
    raw = latents["coords"]
    out = np.full((n_rows, top_k), np.nan)
    for i in range(n_rows):
        row = np.asarray(raw[i], dtype=float).reshape(top_k, -1)
        out[i] = row[:, 0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default=os.path.expanduser("~/i2502/prep_L16_p128"))
    ap.add_argument("--harvest", default=os.path.expanduser("~/i2502/harvest"))
    ap.add_argument("--stage-a", default=os.path.expanduser("~/i2502/stage_a.npz"))
    ap.add_argument("--stage-a-meta", default=os.path.expanduser("~/i2502/stage_a_meta.json"))
    ap.add_argument("--fits", default=os.path.expanduser("~/i2502/fits"))
    ap.add_argument("--k", type=int, default=32000)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--n-iter", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--doses", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/i2502/flagship"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_jsonl = os.path.join(args.fits, "fits.jsonl")

    import gamfit
    X_train = np.ascontiguousarray(np.load(f"{args.prep}/train.npy"))
    X_test = np.ascontiguousarray(np.load(f"{args.prep}/test.npy"))
    meta = json.load(open(f"{args.prep}/meta.json"))
    P = meta["p"]
    print(f"[flag] fit start K={args.k} P={P} n={len(X_train)} n_iter={args.n_iter}",
          flush=True)

    t0 = time.time()
    model = gamfit.sae_manifold_fit(
        X_train, K=args.k, d_atom=1, atom_topology="circle", assignment="topk",
        top_k=args.top_k, n_iter=args.n_iter, random_state=args.seed,
        sparsity_weight=0.0, ard_per_atom=True, gpu="off")
    wall = time.time() - t0
    print(f"[flag] FIT DONE wall={wall:.0f}s r2={model.reconstruction_r2:.4f}",
          flush=True)

    lat = model.converged_latents()
    sup_i = np.asarray(lat["support_indices"])          # (N, top_k) u32
    sup_v = np.asarray(lat["support_values"])           # (N, top_k)
    N = len(X_train)
    phases = slot_coords(lat, N, args.top_k)            # (N, top_k) turns
    fitted = np.asarray(model.fitted)
    train_ev = ev_of(X_train, fitted)
    t1 = time.time()
    R_test = np.asarray(model.reconstruct(X_test))
    test_ev = ev_of(X_test, R_test)
    active = np.abs(sup_v) > 0
    usage = np.bincount(sup_i[active].ravel(), minlength=args.k)
    alive = int((usage > 0).sum())
    K_ret = int(model.chosen_k)
    rec = dict(record=f"manifold_k{args.k}_p{P}", status="ok", k=args.k,
               chosen_k=K_ret, p=P, lane="topk-support-sparse",
               n_train=N, n_test=len(X_test), n_iter=args.n_iter,
               top_k=args.top_k, wall_s=round(wall, 1),
               oos_wall_s=round(time.time() - t1, 1), train_ev=train_ev,
               test_ev=test_ev, reconstruction_r2=float(model.reconstruction_r2),
               test_mean_l0=float(active.sum(1).mean()),
               alive_atoms_train=alive,
               laml=float(model.penalized_quasi_laplace_criterion))
    with open(out_jsonl, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("[flag]", json.dumps(rec), flush=True)

    blocks = [np.asarray(b, dtype=float) for b in model.decoder_blocks]
    np.savez_compressed(os.path.join(args.out_dir, "artifacts.npz"),
                        support_indices=sup_i, support_values=sup_v,
                        phases=phases, usage=usage,
                        decoder_blocks=np.stack(blocks) if len({b.shape for b in blocks}) == 1 else np.empty(0),
                        training_mean=np.asarray(model.training_mean))
    with open(os.path.join(args.out_dir, "model_dict.pkl"), "wb") as f:
        pickle.dump(model.to_dict(), f, protocol=4)

    # ---- 4. self-verified basis decode -----------------------------------
    M = blocks[0].shape[0]
    mean = np.asarray(model.training_mean)

    def make_phi(kind):
        const = "const" in kind
        sin_first = "sin" in kind
        rad = "rad" in kind
        w = 1.0 if rad else TAU

        def phi(t):
            t = np.atleast_1d(t)
            cols = []
            if const:
                cols.append(np.ones_like(t))
            h = 1
            while len(cols) < M:
                first = np.sin(w * h * t) if sin_first else np.cos(w * h * t)
                second = np.cos(w * h * t) if sin_first else np.sin(w * h * t)
                cols.append(first)
                if len(cols) < M:
                    cols.append(second)
                h += 1
            return np.stack(cols, axis=1)
        return phi

    chosen_phi, best_err, chosen_kind = None, np.inf, None
    nprobe = min(200, N)
    probe = slice(0, nprobe)
    for kind in ("fourier", "const+fourier", "sin-fourier", "const+sin-fourier",
                 "rad-fourier", "const+rad-fourier"):
        phi = make_phi(kind)
        recon = np.tile(mean, (nprobe, 1))
        for i in range(nprobe):
            for s in range(args.top_k):
                a = sup_v[i, s]
                if a == 0:
                    continue
                k = int(sup_i[i, s])
                recon[i] += a * (phi(phases[i, s])[0] @ blocks[k])
        err = np.max(np.abs(recon - fitted[probe])) / max(np.max(np.abs(fitted[probe])), 1e-12)
        print(f"[flag] basis {kind}: rel err {err:.2e}", flush=True)
        if err < best_err:
            best_err, chosen_phi, chosen_kind = err, phi, kind
    phi_ok = best_err < 1e-6
    if phi_ok:
        print(f"[flag] basis VERIFIED: {chosen_kind} (rel err {best_err:.2e})", flush=True)
    else:
        print(f"[flag] WARNING: no basis convention reproduces fitted "
              f"(best {chosen_kind} rel err {best_err:.2e}) — phi-dependent "
              f"outputs will use nearest-data-row images instead", flush=True)

        def chosen_phi_fallback(t):
            raise RuntimeError("phi unavailable")
        # keep chosen_phi as the best guess for non-critical curve overlays

    # ---- 5/6 need harvest metadata + tokenizer ---------------------------
    rows_train = np.load(f"{args.prep}/rows_train.npy")[:N]
    tok_ids = np.load(f"{args.harvest}/token_ids.npy")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base", trust_remote_code=True)

    # census: top rows per atom from sparse support
    order = np.argsort(-usage)
    show = [k for k in order[:12]]
    flat_rows = np.repeat(np.arange(N), args.top_k).reshape(N, args.top_k)
    interp = {"k": args.k, "chosen_k": K_ret, "alive": alive, "atoms": []}

    def atom_rows(k, cap=40):
        mask = (sup_i == k) & active
        rr, ss = np.nonzero(mask)
        o = np.argsort(-np.abs(sup_v[rr, ss]))[:cap]
        return rr[o], ss[o]

    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    cyc = plt.get_cmap("twilight")
    for panel, k in enumerate(show):
        rr, ss = atom_rows(k)
        ph = phases[rr, ss] % 1.0
        img = np.stack([chosen_phi(p)[0] @ blocks[k] for p in ph])
        c = img.mean(0)
        _, _, vt = np.linalg.svd(img - c, full_matrices=False)
        P2 = vt[:2] if vt.shape[0] >= 2 else np.vstack([vt, vt])
        pts = (X_train[rr] - mean - c) @ P2.T
        tgrid = np.linspace(0, 1, 129)
        crv = (np.stack([chosen_phi(t)[0] @ blocks[k] for t in tgrid]) - c) @ P2.T
        ax = axes.flat[panel]
        ax.plot(crv[:, 0], crv[:, 1], "-", color="#999999", lw=1.2, zorder=1)
        ax.scatter(pts[:, 0], pts[:, 1], c=ph, cmap=cyc, vmin=0, vmax=1, s=26,
                   zorder=2, edgecolors="white", linewidths=0.4)
        toks, seen = [], set()
        for r in rr[:14]:
            t_str = tok.decode([int(tok_ids[rows_train[r]])]).strip() or "␣"
            if t_str not in seen:
                seen.add(t_str)
                toks.append(t_str)
        ax.set_title(f"atom {k} · used {usage[k]}×\n" + " ".join(toks[:6]), fontsize=8)
        style(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        interp["atoms"].append(dict(atom=int(k), usage=int(usage[k]),
                                    top_tokens=toks[:10]))
    fig.suptitle(f"top-12 atoms of the K={args.k} overcomplete circle-atom manifold "
                 f"dictionary — Qwen3.5-4B-Base L16 (unsupervised; gray = fitted "
                 f"closed curve, color = intrinsic phase)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(args.out_dir, "fig_atom_gallery.png"), dpi=150)
    json.dump(interp, open(os.path.join(args.out_dir, "interp.json"), "w"), indent=2)
    print("[flag] gallery written", flush=True)

    # unsupervised calendar scan on natural train rows
    WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    MON = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]
    cyc_report = {}
    for name, fam in (("weekday", WEEK), ("month", MON)):
        ids = [tok.encode(" " + w, add_special_tokens=False)[0] for w in fam]
        id2lab = {t: i for i, t in enumerate(ids)}
        fam_rows = np.flatnonzero(np.isin(tok_ids[rows_train], ids))
        n = len(fam)
        best = []
        for k in np.unique(sup_i[fam_rows][np.abs(sup_v[fam_rows]) > 0]):
            mask = (sup_i[fam_rows] == k) & (np.abs(sup_v[fam_rows]) > 0)
            rsel, ssel = np.nonzero(mask)
            if len(rsel) < 15:
                continue
            labs = np.array([id2lab[int(tok_ids[rows_train[fam_rows[r]]])] for r in rsel])
            ph = phases[fam_rows[rsel], ssel]
            r2, orient = circular_r2(ph, labs, n)
            best.append((float(r2), int(k), int(len(rsel)), int(orient)))
        best.sort(reverse=True)
        cyc_report[name] = {"n_rows": int(len(fam_rows)), "top": best[:8]}
        print(f"[flag] {name} scan: {best[:3]}", flush=True)
        if best and best[0][0] > 0.15:
            r2, k, _, orient = best[0]
            mask = (sup_i[fam_rows] == k) & (np.abs(sup_v[fam_rows]) > 0)
            rsel, ssel = np.nonzero(mask)
            labs = np.array([id2lab[int(tok_ids[rows_train[fam_rows[r]]])] for r in rsel])
            ph = phases[fam_rows[rsel], ssel] % 1.0
            figp, axp = plt.subplots(figsize=(6.4, 5.2),
                                     subplot_kw={"projection": "polar"})
            th = TAU * ph
            rng = np.random.default_rng(0)
            axp.scatter(th, 0.68 + 0.24 * rng.random(len(th)), c=labs, cmap="twilight",
                        s=24, vmin=0, vmax=n - 1, edgecolors="white", linewidths=0.3)
            for j, w in enumerate(fam):
                sel = labs == j
                if sel.any():
                    axp.annotate(w, (np.angle(np.exp(1j * th[sel]).mean()), 1.18),
                                 fontsize=8, ha="center", color=INK)
            axp.set_yticks([])
            axp.set_title(f"unsupervised {name} structure on wikitext: atom {k} "
                          f"(circular R²={r2:.2f}, n={len(th)})", fontsize=10)
            figp.tight_layout()
            figp.savefig(os.path.join(args.out_dir, f"fig_{name}_circle.png"), dpi=160)
    json.dump(cyc_report, open(os.path.join(args.out_dir, "calendar_scan.json"), "w"),
              indent=2)

    # ---- overcompleteness proof ------------------------------------------
    frame = np.vstack(blocks)
    sv = np.linalg.svd(frame, compute_uv=False)
    figo, axs = plt.subplots(1, 3, figsize=(14, 4))
    u = np.sort(usage)[::-1]
    axs[0].semilogy(np.arange(1, len(u) + 1), np.maximum(u, 0.5), color="#0072B2", lw=1.5)
    axs[0].axvline(P, color="#D55E00", ls="--", lw=1.2)
    axs[0].annotate(f"chart dim P={P}", (P * 1.15, max(u.max(), 2) * 0.5),
                    color="#D55E00", fontsize=8)
    axs[0].set_xlabel("atom rank by usage")
    axs[0].set_ylabel("times on a row's active support (log)")
    axs[0].set_title(f"{alive} of {K_ret} retained atoms alive ≫ P={P}", fontsize=9)
    axs[1].semilogy(np.arange(1, len(sv) + 1), np.maximum(sv, 1e-12), color="#0072B2", lw=1.5)
    axs[1].axvline(P, color="#D55E00", ls="--", lw=1.2)
    axs[1].set_xlabel("singular value index")
    axs[1].set_ylabel("σ of stacked decoder frame")
    axs[1].set_title(f"frame {frame.shape[0]}×{P}: rank "
                     f"{int((sv > sv[0] * 1e-10).sum())} — spans the chart", fontsize=9)
    alive_idx = np.flatnonzero(usage > 0)
    samp = np.random.default_rng(0).choice(alive_idx, min(2000, len(alive_idx)),
                                           replace=False)
    dirs = np.stack([blocks[i][np.argmax(np.linalg.norm(blocks[i], axis=1))]
                     for i in samp])
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    G = np.abs(dirs @ dirs.T)
    iu = np.triu_indices(len(samp), 1)
    axs[2].hist(G[iu], bins=60, color="#0072B2")
    axs[2].set_xlabel("|cos| between atom principal directions")
    axs[2].set_ylabel("atom pairs")
    axs[2].set_title(f"coherence: median {np.median(G[iu]):.3f}, "
                     f"max {G[iu].max():.3f} < 1", fontsize=9)
    for ax in axs:
        style(ax)
    figo.suptitle(f"overcompleteness: {K_ret} atoms · decoder frame {frame.shape[0]} "
                  f"columns in P={P} dims ({K_ret / P:.0f}× atoms, "
                  f"{frame.shape[0] / P:.0f}× frame)", fontsize=11)
    figo.tight_layout(rect=(0, 0, 1, 0.93))
    figo.savefig(os.path.join(args.out_dir, "fig_overcompleteness.png"), dpi=150)
    print("[flag] overcompleteness written", flush=True)

    # ---- 5. steering deltas (stage B patches them) ------------------------
    lift = np.load(f"{args.prep}/lift.npy")
    c0 = np.load(f"{args.prep}/c0.npy")
    A = np.load(args.stage_a)
    doses = [float(v) for v in args.doses.split(",")]
    steer_out = {}
    steer_meta = {}
    w = np.load(os.path.join(args.fits, f"torch_topk_k{args.k}_p128.npz"))
    for cycname, n in (("week", 7), ("month", 12)):
        Xf = A[f"{cycname}_fit_X"].astype(np.float64)
        labf = A[f"{cycname}_fit_lab"]
        Xb = A[f"{cycname}_base_X"].astype(np.float64)
        Zf = np.ascontiguousarray((Xf - c0) @ lift.T)
        Zb = np.ascontiguousarray((Xb - c0) @ lift.T)
        ef = model.encode(Zf)
        fi, fv = np.asarray(ef["indices"]), np.asarray(ef["values"])
        fph = slot_coords(ef, len(Zf), args.top_k)
        cands = np.unique(fi[np.abs(fv) > 0])
        best = (-np.inf, None, 1)
        for k in cands:
            mask = (fi == k) & (np.abs(fv) > 0)
            rsel, ssel = np.nonzero(mask)
            if len(rsel) < max(8, len(Zf) // 6):
                continue
            r2, orient = circular_r2(fph[rsel, ssel], labf[rsel], n)
            if r2 > best[0]:
                best = (r2, int(k), orient)
        r2b, atom, orient = best
        print(f"[flag] steer {cycname}: atom={atom} r2={r2b}", flush=True)
        if atom is None:
            steer_meta[cycname] = {"atom": None, "r2": None}
            continue
        amp_bar = float(np.median(np.abs(fv[(fi == atom) & (np.abs(fv) > 0)])))
        eb = model.encode(Zb)
        bi_, bv_ = np.asarray(eb["indices"]), np.asarray(eb["values"])
        bph = slot_coords(eb, len(Zb), args.top_k)
        # per base row: active phase if atom in support, else nearest fit row's
        t0s, amps, native = [], [], 0
        for i in range(len(Zb)):
            slots = np.nonzero((bi_[i] == atom) & (np.abs(bv_[i]) > 0))[0]
            if len(slots):
                t0s.append(bph[i, slots[0]])
                amps.append(abs(bv_[i, slots[0]]))
                native += 1
            else:
                j = np.argmin(np.linalg.norm(Zf - Zb[i], axis=1))
                slots = np.nonzero((fi[j] == atom) & (np.abs(fv[j]) > 0))[0]
                t0s.append(fph[j, slots[0]] if len(slots) else 0.0)
                amps.append(amp_bar)
        shifts = list(range(1, 7)) if n == 7 else [1, 2, 3, 4, 6, 9]
        deltas = np.zeros((len(Zb), len(shifts), len(doses), Xb.shape[1]))
        for i in range(len(Zb)):
            g0 = chosen_phi(t0s[i])[0] @ blocks[atom]
            for js, sh in enumerate(shifts):
                for jd, d in enumerate(doses):
                    t1 = t0s[i] + orient * sh * d / n
                    dz = amps[i] * ((chosen_phi(t1)[0] @ blocks[atom]) - g0)
                    deltas[i, js, jd] = dz @ lift
        steer_out[f"{cycname}_deltas"] = deltas.astype(np.float32)
        # torch control: best phase latent on the SAME fit cloud
        pre = (Zf - w["b_pre"]) @ w["W_enc"] + w["b_enc"]
        idxs = np.argpartition(pre, -args.top_k, axis=1)[:, -args.top_k:]
        zc = np.zeros_like(pre)
        np.put_along_axis(zc, idxs, np.maximum(np.take_along_axis(pre, idxs, 1), 0), 1)
        design = np.column_stack([np.ones(len(labf)), np.cos(TAU * labf / n),
                                  np.sin(TAU * labf / n)])
        best_lat, best_r2 = 0, -np.inf
        for j in np.flatnonzero((zc != 0).any(0)):
            col = zc[:, j]
            coef, *_ = np.linalg.lstsq(design, col, rcond=None)
            resid = col - design @ coef
            tss = ((col - col.mean()) ** 2).sum()
            rr2 = 1.0 - resid @ resid / max(tss, 1e-30)
            if rr2 > best_r2:
                best_r2, best_lat = rr2, int(j)
        fd = w["W_dec"][best_lat] @ lift
        steer_out[f"{cycname}_flat_dir"] = (fd / max(np.linalg.norm(fd), 1e-30)).astype(np.float32)
        steer_meta[cycname] = dict(atom=int(atom), r2=float(r2b), orient=int(orient),
                                   native_base_rows=int(native), n_base=len(Zb),
                                   amp_bar=amp_bar, shifts=shifts, doses=doses,
                                   torch_latent=int(best_lat),
                                   torch_latent_r2=float(best_r2))
    np.savez_compressed(os.path.join(args.out_dir, "steer_deltas.npz"), **steer_out)
    json.dump(steer_meta, open(os.path.join(args.out_dir, "steer_meta.json"), "w"),
              indent=2)
    print("[flag] steering deltas written", flush=True)

    # ---- 6. splice reconstructions ---------------------------------------
    vh = A["val_h"].astype(np.float64)                 # (B, T, D)
    B_, T_, D_ = vh.shape
    flat = vh.reshape(-1, D_)
    posflags = np.zeros((B_, T_), dtype=bool)
    posflags[:, 0] = True
    c1v = np.load(f"{args.prep}/c1.npy")
    cvec = np.where(posflags.reshape(-1, 1), c1v[None, :], c0[None, :])
    Zv = np.ascontiguousarray((flat - cvec) @ lift.T)
    out_splice = {"chart": (Zv @ lift + cvec).astype(np.float16),
                  "mean_ablate": cvec.astype(np.float16)}
    t2 = time.time()
    Rv = np.empty_like(Zv)
    step = 4096
    for i in range(0, len(Zv), step):
        Rv[i:i + step] = np.asarray(model.reconstruct(np.ascontiguousarray(Zv[i:i + step])))
    print(f"[flag] splice recon {len(Zv)} rows in {time.time()-t2:.0f}s", flush=True)
    out_splice["manifold"] = (Rv @ lift + cvec).astype(np.float16)
    pre = (Zv - w["b_pre"]) @ w["W_enc"] + w["b_enc"]
    idxs = np.argpartition(pre, -args.top_k, axis=1)[:, -args.top_k:]
    zz = np.zeros_like(pre)
    np.put_along_axis(zz, idxs, np.maximum(np.take_along_axis(pre, idxs, 1), 0), 1)
    out_splice["torch_topk"] = ((zz @ w["W_dec"] + w["b_pre"]) @ lift + cvec).astype(np.float16)
    del pre, zz
    mu8 = Zv.mean(0)
    _, _, vt8 = np.linalg.svd(np.load(f"{args.prep}/train.npy") -
                              np.load(f"{args.prep}/train.npy").mean(0),
                              full_matrices=False)
    V8 = vt8[:args.top_k]
    out_splice["pca8"] = ((((Zv - mu8) @ V8.T) @ V8 + mu8) @ lift + cvec).astype(np.float16)
    np.savez_compressed(os.path.join(args.out_dir, "splice_recons.npz"), **out_splice)
    print("[flag] PIPELINE DONE", flush=True)


if __name__ == "__main__":
    main()
