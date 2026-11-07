#!/usr/bin/env python3
"""Score greedy Grounder outputs on Test2445 with the unmodified official eval_script. CPU only (da_env2).

Each --states file is a generate_states.py output (pred = the greedy set E). Writes one eval_script pred dir per
row and prints, macro over the 6 domains: GroupIoU and F1@50 per domain and macro, from the official script
(F1@50 there is many-to-many, R and P averaged per domain), plus the 1:1 Hungarian F1@0.5 (report.py) for the
P/R definition the paper uses. Both on all 2,445 (the 13 items without gold count as 0, as eval_script does) and
on the scorable 2,432.

    python dragon_grpo_r1/paper/score_test.py --states sft=TEST/sft.jsonl --states grpo=TEST/grpo.jsonl \\
        --scorable data_paper/test2432_ids.txt --out-dir TEST/report_main
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

R1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R1 / "completer_pilot"))
import report as RP  # noqa: E402
from gc_engine import VC  # noqa: E402

DOMS = ["ai2d", "chartqa", "circuitvqa", "infographics", "mapiq", "mapwise"]


def score_sets(name, recs, sets, out):
    """Official eval on (records, predicted sets); returns {'macro': .., 'per_domain': .., 'f1_11': ..}."""
    d = out / "preds" / name
    VC.write_pred_dir(d, [{"id": r["id"], "domain": r["domain"], "gold": r["gold"], "pred": s,
                           "response": r.get("E_response", ""), "n_tokens": r.get("E_tokens", 0)}
                          for r, s in zip(recs, sets)])
    res = VC.run_official_eval(d)
    items = [{"id": r["id"], "domain": r["domain"], "gold": r["gold"]} for r in recs]
    res["f1_11"] = RP.Row(name, items, sets).m["F1_1:1"]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--scorable", required=True, help="ids of the 2,432 items with gold")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    log = RP.make_logger(out / "report.txt")
    keep = set(Path(a.scorable).read_text().split())
    table = {}
    for spec in a.states:
        name, path = spec.split("=", 1)
        recs = RP.load_jsonl(path)
        n = len(recs)
        if n != 2445:
            log(f"WARNING: {name} has {n} items, not 2,445")
        sub = [r for r in recs if r["id"] in keep]
        table[name] = {"2445": score_sets(name, recs, [r["E"] for r in recs], out),
                       "2432": score_sets(name + "_2432", sub, [r["E"] for r in sub], out)}
    for split in ["2445", "2432"]:
        log(f"\n===== Test{split} (official eval_script; macro over domains) =====")
        log(f"{'row':>8} " + " ".join(f"{d[:10]:>10}" for d in DOMS) +
            f" {'GroupIoU':>9} {'F1@50':>7} {'F1 1:1':>7} {'MeanIoU':>8}")
        for name, t in table.items():
            r = t[split]
            log(f"{name:>8} " + " ".join(f"{r['per_domain'][d]['GroupIoU']:10.4f}" for d in DOMS) +
                f" {r['macro']['GroupIoU']:9.4f} {r['macro']['F1@50']:7.4f} {r['f1_11']:7.4f} {r['macro']['MeanIoU']:8.4f}")
        log("  per-domain F1@50: " + "; ".join(
            f"{name}: " + ", ".join(f"{d} {t[split]['per_domain'][d]['F1@50']:.4f}" for d in DOMS)
            for name, t in table.items()))
    (out / "scores.json").write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
