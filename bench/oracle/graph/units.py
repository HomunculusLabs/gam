"""Unit circuits, the oracle's answer measured by the carry test (#2951 graph oracle).

An answer is plain text (nothing in it is executed): an English explanation, then one line per input position of the
text, the units through which a change of that position's token reaches the prediction:

  EXPLANATION: the model copies the digits after the earlier "327" ...
  POSITION 322: L0.fc@322 L0.down@322 L2.v@322[h3,h4] L2.o@332[h3,h4]

A unit is a weight matrix of a layer at a position (L<layer>.<q|k|v|o|fc|down>@<position>), for attention optionally
some of its heads ([h<i>,...]: the head's slice of the activation, carry.UnitSpace). An answer is measured by the
carry test (carry.py): for each of the text's K positions where a changed token moves the prediction most
(Native.sensitivity), the prompts changing that token to the model's 3 likeliest others there (carry.alternatives),
every unit outside the answer's line for that position held at its value on the text and the line's units computing
on the changed prompt; the KL of the model on the changed prompt from that run, against the KL with no units (how far
the model moves). Its score: the share of the movement it leaves uncarried, the positions weighted by their movement
(a position without a line carries nothing); its size: the number of units (a head counts as one).

  units.py score ANSWER.txt TASK_ID [--k 8]
  units.py baseline copy|all TASK_ID [--k 8]                  a search-free answer's score
  units.py targets EXPLAIN.jsonl OUT_DIR [--fraction 90]     answers from carry.py explain's measured circuits
  units.py dataset EXPLAIN.jsonl OUT.jsonl [--fraction 90]   SFT rows (question, measured answer)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

KINDS = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj", "fc": "c_fc", "down": "down_proj"}
CODES = {v: k for k, v in KINDS.items()}
UNIT = re.compile(r"L(\d+)\.(q|k|v|o|fc|down)@(\d+)(?:\[([h\d,\s]+)\])?")
LINE = re.compile(r"^\s*POSITION\s+(\d+)\s*:(.*)$", re.M)


def parse(text: str) -> tuple[str, dict]:
    """(the explanation, {position: [unit, ...]}) of an answer; a unit is (matrix name, position) or (matrix name,
    head, position). Lines that name no unit are kept empty; text that is not a unit is ignored."""
    import native

    m = re.search(r"EXPLANATION:\s*(.*?)(?=^\s*POSITION\s|\Z)", text, re.S | re.M)
    explanation = m[1].strip() if m else ""
    circuits = {}
    for line in LINE.finditer(text):
        units = []
        for u in UNIT.finditer(line[2]):
            name = native.site_name(int(u[1]), KINDS[u[2]])
            heads = [int(h) for h in re.findall(r"\d+", u[4])] if u[4] else None
            units += [(name, h, int(u[3])) for h in heads] if heads and ".attn." in name else [(name, int(u[3]))]
        circuits.setdefault(int(line[1]), []).extend(units)
    return explanation, circuits


def render(explanation: str, circuits: dict) -> str:
    """The answer text of circuits {position: [unit, ...]} (heads of one matrix and position written together)."""
    out = [f"EXPLANATION: {explanation}".rstrip()]
    for p in sorted(circuits, key=lambda q: -1 if q is None else q):
        whole, heads = [], {}
        for u in circuits[p]:
            if len(u) == 3:
                heads.setdefault((u[0], u[2]), []).append(u[1])
            else:
                whole.append(u)
        words = []
        for name, pos in sorted(whole, key=lambda u: (int(u[0].split(".")[1]), u[1], u[0])):
            layer = int(name.split(".")[1])
            words.append(f"L{layer}.{CODES[name.split('.')[-1]]}@{pos}")
        for (name, pos), hs in sorted(heads.items(), key=lambda kv: (int(kv[0][0].split(".")[1]), kv[0][1], kv[0][0])):
            layer = int(name.split(".")[1])
            words.append(f"L{layer}.{CODES[name.split('.')[-1]]}@{pos}[{','.join(f'h{h}' for h in sorted(hs))}]")
        out.append(f"POSITION {p}: " + " ".join(words))
    return "\n".join(out)


def _active(space, units):
    """The space's coefficient vector with these units on (a whole attention matrix in a head space: all its heads)."""
    import torch

    on = torch.zeros(space.size, dtype=torch.bool, device=space.nat.dev)
    want = set(units)
    for j, u in enumerate(space.units):
        if u in want or (len(u) == 3 and (u[0], u[2]) in want):
            on[j] = True
    return on


def sensitive(nat, ids: list[int], targets: list[int], k: int) -> list[int]:
    """The k positions (not the targets) whose changed token moves the prediction most, most first: carry.explain's
    choice (Native.sensitivity under the text's seed)."""
    import torch

    import native

    if max(targets) < 1:
        return []
    with torch.no_grad():
        moved, _ = nat.sensitivity(ids, targets, torch.Generator(device="cpu").manual_seed(native.task_seed(str(ids[:8]))))
    return [p for p in (moved.argsort(descending=True, stable=True) + 1).tolist() if p not in targets][:k]


def score(nat, task: dict, text: str, k: int = 8) -> dict:
    """The carry test of an answer on a task: {"uncarried": the movement-weighted share left uncarried, "units", per
    position {"position", "movement" (bits), "carried" (share), "units", "outside" (units not at the position or the
    targets, which the test cannot place)}, "explanation"}."""
    import torch

    import carry
    import native

    explanation, circuits = parse(text)
    prompt = task["prompts"][0]
    ids, targets = prompt["token_ids"], prompt["target_positions"]
    positions = sensitive(nat, ids, targets, k)
    rows, total_w, total_u = [], 0.0, 0.0
    for p in positions:
        prompts = carry.alternatives(nat, ids, p)
        stated = circuits.get(p, [])
        inside = [u for u in stated if u[-1] in (p, *targets)]
        gran = "head" if any(len(u) == 3 for u in inside) else "matrix"
        space = carry.UnitSpace(nat, len(ids), [p, *targets], gran)
        study = (carry.HeadCarry if gran == "head" else carry.MatrixCarry)(nat, ids, targets, prompts, space)
        with torch.no_grad():
            empty, with_units = study.measure([_active(space, []), _active(space, inside)]).tolist()
        carried = 0.0 if empty <= 0 else max(0.0, min(1.0, 1 - with_units / empty))
        rows.append({"position": p, "movement": empty, "carried": carried, "units": len(inside), "outside": len(stated) - len(inside)})
        total_w += empty
        total_u += empty * (1 - carried)
        native.free(nat.dev)
    return {"uncarried": total_u / total_w if total_w else 0.0, "units": sum(r["units"] for r in rows), "positions": rows,
            "explanation": explanation}


def targets(explain_jsonl: Path, out_dir: Path, fraction: int = 90) -> int:
    """One answer per question of carry.py explain's output: for each position, the smallest measured unit set on its
    elimination curve that carries `fraction`% of what the position's full unit set carries (a position whose full set
    carries less than all of its movement needs units elsewhere too; its line then states the part that is at the
    position and the targets), no explanation (the English is trained by its own reader). Positions whose full set
    carries nothing get no line."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for line in open(explain_jsonl):
        r = json.loads(line)
        circuits = {}
        for p in r["positions"]:
            curve, sets, empty = p["curve"], p["unit_sets"], p["empty_bits"]
            full = min(kl for _, kl in curve)
            if empty <= 0 or full >= empty:
                continue
            goal = empty - fraction / 100 * (empty - full)
            fits = [i for i, (_, kl) in enumerate(curve) if kl <= goal]
            best = sets[max(fits, key=lambda i: -curve[i][0])]  # the fewest units within the goal
            circuits[p["position"]] = [tuple(u) for u in best]
        (out_dir / f"{r['task']}.txt").write_text(render("", circuits))
        n += 1
    return n


ASK = ("How does {model} compute its prediction of the next token after the target of this text? For each listed input "
       "position, name the units through which a change of the token there reaches the prediction: a unit is a weight "
       "matrix of a layer at a position, L<layer>.<q|k|v|o|fc|down>@<position>, an attention matrix optionally by its "
       "heads, L<layer>.<q|k|v|o>@<position>[h<i>,...]. Answer with a line EXPLANATION: (in plain English, what the model "
       "does), then one line POSITION <p>: <units> per listed position. Fewer units are better, as long as they carry "
       "the change.")


def question(task: dict, positions: list[int]) -> str:
    """The oracle's question for unit circuits: the ask, the text as tokens with the target and the model's most
    probable next tokens (prompt.line), and the positions whose changed token moves the prediction most."""
    from prompt import line

    model = task["model"]
    p = task["prompts"][0]
    return "\n".join([ASK.format(model=model), line(model, p["token_ids"], p["target_positions"], p["model_top"]),
                      "positions where a changed token moves the prediction most, most first: " + ", ".join(map(str, positions))])


def baseline(kind: str, positions: list[int], targets: list[int]) -> str:
    """Answers that need no search, for comparison: "copy", the recurring copy circuit at every position (layer-0 MLP
    at the position, layer-2 values there, layer-2 output at the target); "all", every unit at the position and the
    target."""
    import native

    t = max(targets)
    circuits = {}
    for p in positions:
        if kind == "copy":
            circuits[p] = [(native.site_name(0, "c_fc"), p), (native.site_name(0, "down_proj"), p), (native.site_name(2, "v_proj"), p),
                           (native.site_name(2, "o_proj"), t)]
        else:
            circuits[p] = [(native.site_name(L, k), q) for L in range(4) for k in KINDS.values() for q in (p, t)]
    return render("", circuits)


def dataset(explain_jsonl: Path, out: Path, fraction: int = 90) -> int:
    """SFT rows {"prompt": [user message], "completion": [assistant message], "info": {"task"}} from carry.py
    explain's measured circuits (targets()'s answers) and their questions."""
    import tempfile

    import native

    with tempfile.TemporaryDirectory() as tmp:
        targets(explain_jsonl, Path(tmp), fraction)
        n = 0
        with open(out, "w") as f:
            for line in open(explain_jsonl):
                r = json.loads(line)
                task = json.loads((native.TEXTS / "vpd4l" / f"{r['task']}.json").read_text())
                answer = (Path(tmp) / f"{r['task']}.txt").read_text()
                positions = [p["position"] for p in r["positions"]]
                f.write(json.dumps({"prompt": [{"role": "user", "content": question(task, positions)}],
                                    "completion": [{"role": "assistant", "content": answer}], "info": {"task": r["task"]}}) + "\n")
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("answer", type=Path)
    s.add_argument("task")
    s.add_argument("--k", type=int, default=8)
    d = sub.add_parser("dataset")
    d.add_argument("explain", type=Path)
    d.add_argument("out", type=Path)
    d.add_argument("--fraction", type=int, default=90)
    b = sub.add_parser("baseline")
    b.add_argument("kind", choices=("copy", "all"))
    b.add_argument("task")
    b.add_argument("--k", type=int, default=8)
    t = sub.add_parser("targets")
    t.add_argument("explain", type=Path)
    t.add_argument("out", type=Path)
    t.add_argument("--fraction", type=int, default=90)
    a = ap.parse_args()
    if a.cmd == "targets":
        print(f"{targets(a.explain, a.out, a.fraction)} answers in {a.out}")
        return
    if a.cmd == "dataset":
        print(f"{dataset(a.explain, a.out, a.fraction)} rows in {a.out}")
        return
    import native

    nat = native.Native()
    task = json.loads((native.TEXTS / "vpd4l" / f"{a.task}.json").read_text())
    if a.cmd == "baseline":
        p = task["prompts"][0]
        text = baseline(a.kind, sensitive(nat, p["token_ids"], p["target_positions"], a.k), p["target_positions"])
    else:
        text = a.answer.read_text()
    print(json.dumps(score(nat, task, text, a.k), indent=1))


if __name__ == "__main__":
    main()
