"""Gather every #2502 result artifact into one JSON for the close comment."""
import json, os, glob

H = os.path.expanduser("~")
out = {}
recs = [json.loads(x) for x in open(f"{H}/i2502/fits/fits.jsonl")]
keep = {}
for r in recs:
    if r.get("status") == "ok":
        keep[r["record"]] = r
out["fits"] = {k: v for k, v in keep.items() if any(
    s in k for s in ("manifold_k32000", "torch_topk_k32000_p128", "pca_M8_p128",
                     "pca_M16_p128", "pca_M64_p128", "linear_k32000",
                     "pilot_topk_k256"))}
for name in ("splice_results", "steer_summary", "steer_meta", "calendar_scan"):
    p = f"{H}/i2502/flagship/{name}.json"
    if os.path.exists(p):
        out[name] = json.load(open(p))
p = f"{H}/i2502/flagship/interp.json"
if os.path.exists(p):
    d = json.load(open(p))
    out["interp_summary"] = dict(k=d.get("k"), chosen_k=d.get("chosen_k"),
                                 alive=d.get("alive"),
                                 atoms=[{kk: a[kk] for kk in ("atom", "usage", "top_tokens")}
                                        for a in d.get("atoms", [])])
out["figures"] = sorted(os.path.basename(f) for f in
                        glob.glob(f"{H}/i2502/flagship/fig_*.png"))
print(json.dumps(out, indent=2)[:6000])
json.dump(out, open(f"{H}/i2502/flagship/all_results.json", "w"), indent=2)
