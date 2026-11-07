#!/usr/bin/env python3
"""v4 Assembler report: one table, three questions, one decision. CPU only (da_env2).

Rows on the same states: G0 greedy E, voting_drop (tuned on TRAIN), v3c (the round-0 v3t Assembler, <select>),
each v4 step, and for the best step the capability ablations computed from the same outputs:
  select-only  fixes back to the original box, no adds        (what verification alone does)
  no-fix       fixes back to the original box, adds kept       (what resize adds)
  no-add       fixes kept, adds dropped                         (what adding adds)
plus any --ablate-sel outputs (blank_visual / blank_sheet at inference, or a trained no-vision control).

Decision (best step chosen on mean(GroupIoU, F1@50) of the steps given):
  1 beats v3c on GroupIoU and F1@50, CI above 0                                    (the headline)
  2 verification: full minus blank_visual (trained control if given, else inference ablation) CI above 0 on GroupIoU
  3 resize: full minus no-fix, CI above 0 on GroupIoU or F1@50
  4 answer-bearing recall above v3c's (items with answer-bearing gold), CI above 0
  5 malformed < 2%

  python completer_pilot/report_v4.py --states S0/val300.jsonl --train-states S0/train.jsonl \\
      --v3c-sel .../sel_step300.jsonl --sel-dir V4/eval --steps 0,100,200,300 \\
      --ablate-sel blank_visual=V4/eval_blank_visual/sel_stepB.jsonl --out-dir V4/report
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gc_common as G  # noqa: E402
import report as RP  # noqa: E402


def v4_sets(items, sel, key="final"):
    by = {s["id"]: s for s in sel}
    return [by[it["id"]][key] if by[it["id"]]["idx"] is not None else list(it["E"]) for it in items]


def ab_recall(items, sets, types):
    """Per-item answer-bearing recall over the items that have answer-bearing gold."""
    return np.array([r for it, S in zip(items, sets)
                     if (r := G.type_recall(S, it["gold"], types.get(it["id"]))) is not None])


def ab_paired(a, b, n_boot, seed=17):
    rng = np.random.default_rng(seed)
    d = b - a
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return dict(gain=float(d.mean()), lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)))


def type_table(items, rows, types, log):
    ts = sorted({t for it in items for t in (types.get(it["id"]) or [])})
    log(f"\nrecall by evidence type (micro over gold boxes)\n{'':>22} " + " ".join(f"{t[:12]:>12}" for t in ts))
    for R in rows:
        hit, tot = Counter(), Counter()
        for it, S in zip(items, R.sets):
            tl = types.get(it["id"]) or []
            h = G.hungarian_hits(S, it["gold"])
            for j, t in enumerate(tl):
                tot[t] += 1; hit[t] += j in h
        log(f"{R.name:>22} " + " ".join(f"{hit[t] / max(1, tot[t]):12.3f}" for t in ts))


def edit_summary(items, sel):
    from prompts_v4 import candidate_scores, parse_v4
    by = {s["id"]: s for s in sel}
    c, acc = Counter(), []
    for it in items:
        s = by[it["id"]]
        if s["idx"] is None:
            c["malformed"] += 1
            continue
        p = parse_v4(s["text"], len(it["crops"]))
        cs = candidate_scores(it["crops"], it["gold"], p)
        acc.append(np.mean(list(cs.values())) if cs else 1.0)
        for d, (x, _) in p["verdict"].items():
            c[x] += 1
            c[f"{x}_ok"] += cs[("v", d)] == 1.0
        c["add"] += len(p["adds"])
        c["add_hit"] += sum(v for k, v in cs.items() if k[0] == "a")
    n = len(items)
    return (f"per item: keep {c['keep'] / n:.2f} drop {c['drop'] / n:.2f} fix {c['fix'] / n:.2f} add {c['add'] / n:.2f} | "
            f"keep right {c['keep_ok'] / max(1, c['keep']):.3f}, drop right {c['drop_ok'] / max(1, c['drop']):.3f}, "
            f"fix lands {c['fix_ok'] / max(1, c['fix']):.3f}, add hits {c['add_hit'] / max(1, c['add']):.3f} | "
            f"verdict credit {np.mean(acc):.3f} | malformed {c['malformed'] / n:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", required=True)
    ap.add_argument("--train-states", required=True)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--prompt", default="v4", choices=list(G.V4_PROMPTS))
    ap.add_argument("--v3c-sel", required=True, help="v3t <select> outputs on the same states (internal indices)")
    ap.add_argument("--sel-dir", required=True)
    ap.add_argument("--steps", default="0,100,200,300")
    ap.add_argument("--ablate-sel", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--types", default="outputs/gc_pilot/data/elements_val600.json")
    ap.add_argument("--subset", default=None, help="jsonl whose ids restrict the items (e.g. Val300-B out of Val600)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    log = RP.make_logger(out / "report.txt")
    recs = RP.load_jsonl(a.states)
    if a.subset:
        keep = {r["id"] for r in RP.load_jsonl(a.subset)}
        recs = [r for r in recs if r["id"] in keep]
    items = [G.completer_item(r, a.K, prompt=a.prompt) for r in recs]
    v3items = [G.completer_item(r, a.K, prompt="v3t") for r in recs]
    types = G.load_gold_types(a.types)
    tr = RP.load_jsonl(a.train_states)
    if len(tr) > 3000:
        tr = [tr[i] for i in np.random.default_rng(0).choice(len(tr), 3000, replace=False)]
    f1 = lambda S, g: (lambda m: 2 * m / (len(S) + len(g)) if S or g else 1.0)(len(G.match_pairs(S, g)))
    kd, _, _ = G.tune_voting(tr, a.K, f1)

    B = RP.Row("G0 greedy", items, [i["E"] for i in items])
    VD = RP.Row(f"voting_drop {kd[0]}/{kd[1]}", items, [G.voting_drop(i["E"], i["pool"], i["e_sup"], *kd) for i in items])
    V3 = RP.Row("v3c", items, RP.final_sets(v3items, RP.load_jsonl(a.v3c_sel)))
    steps, best = {}, None
    for st in a.steps.split(","):
        sel = RP.load_jsonl(Path(a.sel_dir) / f"sel_step{st}.jsonl")
        R = RP.Row(f"v4 step{st}", items, v4_sets(items, sel))
        steps[st] = (R, sel)
        sc = (R.m["GroupIoU"] + R.m["F1@50"]) / 2
        if best is None or sc > best[0]:
            best = (sc, st)
    st = best[1]
    R, sel = steps[st]
    caps = {k: RP.Row(f"v4 step{st} {n}", items, v4_sets(items, sel, k))
            for k, n in [("final_select", "select-only"), ("final_nofix", "no-fix"), ("final_noadd", "no-add")]}
    abl = {}
    for spec in a.ablate_sel:
        name, path = spec.split("=", 1)
        abl[name] = RP.Row(f"v4 {name}", items, v4_sets(items, RP.load_jsonl(path)))
    rows = [B, VD, V3] + [x[0] for x in steps.values()] + list(caps.values()) + list(abl.values())
    RP.table(rows, log)

    log(f"\nvoting_drop tuned on {len(tr)} train states (1:1 F1): k_add={kd[0]} k_keep={kd[1]}")
    log(f"best step {st}: {edit_summary(items, sel)}")
    pv = {"vs v3c": RP.paired(V3, R, a.n_boot), "vs voting_drop": RP.paired(VD, R, a.n_boot),
          "vs select-only (fix + add)": RP.paired(caps["final_select"], R, a.n_boot),
          "vs no-fix (resize)": RP.paired(caps["final_nofix"], R, a.n_boot),
          "vs no-add (add)": RP.paired(caps["final_noadd"], R, a.n_boot),
          "select-only vs v3c (verification alone)": RP.paired(V3, caps["final_select"], a.n_boot)}
    for name, A in abl.items():
        pv[f"vs {name}"] = RP.paired(A, R, a.n_boot)
    for k, v in pv.items():
        log(f"  {k:42s} {RP.fmt_paired(v)}")

    abr = {r.name: ab_recall(items, r.sets, types) for r in [B, VD, V3, R] + list(caps.values()) + list(abl.values())}
    log(f"\nanswer-bearing recall (mean over the {len(abr[R.name])} items that have answer-bearing gold)")
    for k, v in abr.items():
        log(f"  {k:28s} {v.mean():.3f}")
    ab_vs = ab_paired(abr[V3.name], abr[R.name], a.n_boot)
    log(f"  v4 - v3c: {ab_vs['gain']:+.3f} [{ab_vs['lo']:+.3f}, {ab_vs['hi']:+.3f}]")
    type_table(items, [B, V3, R] + list(caps.values()), types, log)

    vis = next((n for n in ("blank_visual_trained", "blank_visual", "blank_sheet") if n in abl), None)
    malformed = float(np.mean([s["idx"] is None for s in sel]))
    checks = {
        "1 beats v3c on GroupIoU and F1@50, CI above 0": all(pv["vs v3c"][k]["lo"] > 0 for k in RP.GATE),
        f"2 verification: full minus {vis or '(no visual ablation given)'} CI above 0 on GroupIoU":
            bool(vis) and pv[f"vs {vis}"]["GroupIoU"]["lo"] > 0,
        "3 resize: full minus no-fix CI above 0 on GroupIoU or F1@50":
            any(pv["vs no-fix (resize)"][k]["lo"] > 0 for k in RP.GATE),
        "4 answer-bearing recall above v3c, CI above 0": ab_vs["lo"] > 0,
        "5 malformed < 2%": malformed < 0.02,
    }
    log(f"\n=== v4 DECISION at best step {st} ===")
    for k, v in checks.items():
        log(f"  {'PASS' if v else 'FAIL'}  {k}")
    log(f"  -> {'WIN' if checks[next(iter(checks))] and checks['5 malformed < 2%'] else 'NO WIN'} on the headline; "
        f"the other checks say which capability carried it")
    (out / "v4_decision.json").write_text(json.dumps({
        "best_step": st, "checks": checks, "paired": pv, "ab_recall": {k: float(v.mean()) for k, v in abr.items()},
        "ab_vs_v3c": ab_vs, "malformed": malformed, "voting_drop_k": list(kd),
        "rows": {r.name: r.m for r in rows}}, indent=1, default=float))


if __name__ == "__main__":
    main()
