#!/usr/bin/env python3
"""Paper tables on Test2445: main table (InternVL rows) and the ablation table. CPU only (da_env2).

Every row is scored by the unmodified official eval_script (GroupIoU and F1@50 per domain, macro over the 6
domains, all 2,445 items; the 13 without gold count 0 for every row). Paired 95% CIs (item bootstrap, 2,000) of
Ours vs voting_drop (the strongest same-compute baseline) and vs GRPO. P and R are per-item 1:1 Hungarian at IoU 0.5.

Main table   SFT | SFT + GRPO | + voting_drop (K = 8) | + Ours (K = 8)
Ablations    A0 GRPO greedy | A1 voting_drop | A2 untrained verifier | A3 warm-up only | A4 flat credit |
             A5 no vision | A6 Ours   (same Grounder, same pool; only the verifier changes)
Rows whose outputs do not exist yet are listed as missing.

    /mnt/data2/traviku2/da_env2/bin/python dragon_grpo_r1/paper/report_paper.py --out-dir outputs/paper/test2445/report_paper
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

R1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R1 / "completer_pilot"))
sys.path.insert(0, str(R1 / "paper"))
import gc_common as G  # noqa: E402
import report as RP  # noqa: E402
import report_v4 as R4  # noqa: E402
from score_test import DOMS, score_sets  # noqa: E402

OUT = R1 / "outputs/paper"
I35 = Path("/mnt/data2/traviku2/dragon_sft_grpo_internvl35_old")


def kept_share(items, sets):
    """Share of the Grounder's correct boxes (E boxes matched 1:1 to gold at IoU >= 0.5) still in the final set."""
    k = n = 0
    for it, S in zip(items, sets):
        for i in G.match_pairs(it["E"], it["gold"]):
            n += 1
            k += any(G.iou(it["E"][i], s) > 0.99 for s in S)
    return k / max(1, n)


def tune_voting(train_states, K):
    tr = RP.load_jsonl(train_states)
    tr = [tr[i] for i in np.random.default_rng(0).choice(len(tr), min(3000, len(tr)), replace=False)]
    f1 = lambda S, g: (lambda m: 2 * m / (len(S) + len(g)) if S or g else 1.0)(len(G.match_pairs(S, g)))
    return G.tune_voting(tr, K, f1)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-dir", default=str(OUT / "test2445"))
    ap.add_argument("--train-states", default=str(OUT / "states/train.jsonl"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    T, out = Path(a.test_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = RP.make_logger(out / "report.txt")
    recs = RP.load_jsonl(T / "grpo.jsonl")
    items = [G.completer_item(r, 8, prompt="v4t") for r in recs]
    if len(items) != 2445:
        log(f"WARNING: {len(items)} Test items, not 2,445")
    kd = tune_voting(a.train_states, 8)
    log(f"voting_drop tuned on train states (1:1 F1): k_add={kd[0]} k_keep={kd[1]}")

    def v4(path):
        return R4.v4_sets(items, RP.load_jsonl(path)) if Path(path).exists() else None

    sft = {r["id"]: r["E"] for r in RP.load_jsonl(T / "sft.jsonl")} if (T / "sft.jsonl").exists() else None
    E = T / "eval"
    rows = {  # name -> (sets, generations per item)
        "SFT": ([sft[i["id"]] for i in items] if sft else None, 1),
        "A0 SFT+GRPO": ([i["E"] for i in items], 1),
        "A1 +voting_drop": ([G.voting_drop(i["E"], i["pool"], i["e_sup"], *kd) for i in items], 9),
        "A2 untrained verifier": (v4(E / "untrained/sel_step0.jsonl"), 10),
        "A3 warm-up only": (v4(E / "ours/sel_warmup.jsonl"), 10),
        "A4 flat credit": (v4(E / "flat/sel_step400.jsonl"), 10),
        "A5 no vision": (v4(E / "novis/sel_step400.jsonl"), 10),
        "A6 Ours": (v4(E / "ours/sel_step400.jsonl"), 10),
    }
    res, R = {}, {}
    for name, (sets, gen) in rows.items():
        if sets is None:
            log(f"  missing: {name}")
            continue
        res[name] = score_sets(name.replace(" ", "_").replace("+", ""), recs, sets, out)
        res[name]["gen"] = gen
        R[name] = RP.Row(name, items, sets)
        res[name]["P"], res[name]["R"] = R[name].m["P_1:1"], R[name].m["R_1:1"]
        res[name]["kept"] = kept_share(items, sets)

    def line(name):
        r = res[name]
        return (f"{name:>22} " + " ".join(f"{r['per_domain'][d]['GroupIoU']:7.3f}" for d in DOMS) +
                f" {r['macro']['GroupIoU']:8.3f} {r['macro']['F1@50']:7.3f} {r['gen']:5d}")

    head = f"{'':>22} " + " ".join(f"{d[:7]:>7}" for d in DOMS) + f" {'GroupIoU':>8} {'F1@50':>7} {'gen':>5}"
    log("\n=== Main table, InternVL3.5-8B, Test2445 (GroupIoU per domain; macro GroupIoU, macro F1@50) ===")
    log(head)
    for n in ["SFT", "A0 SFT+GRPO", "A1 +voting_drop", "A6 Ours"]:
        if n in res:
            log(line(n))
    log("\n=== Ablations (same Grounder and pool; only the verifier changes) ===")
    log(f"{'':>22} {'P':>6} {'R':>6} {'kept':>6} {'GroupIoU':>8} {'F1@50':>7}")
    for n in [k for k in rows if k.startswith("A")]:
        if n in res:
            r = res[n]
            log(f"{n:>22} {r['P']:6.3f} {r['R']:6.3f} {r['kept']:6.3f} {r['macro']['GroupIoU']:8.3f} {r['macro']['F1@50']:7.3f}")
    log("  P, R: per-item 1:1 Hungarian at IoU 0.5; kept: share of the Grounder's correct boxes still in the final set")

    if "A6 Ours" in R:
        log("\n=== Paired 95% CIs (item bootstrap) ===")
        for base in ["A1 +voting_drop", "A0 SFT+GRPO", "A2 untrained verifier", "A3 warm-up only", "A4 flat credit",
                     "A5 no vision"]:
            if base in R:
                p = RP.paired(R[base], R["A6 Ours"], a.n_boot)
                log(f"  Ours - {base:22s} " + "  ".join(
                    f"{k} {p[k]['gain']:+.3f} [{p[k]['lo']:+.3f}, {p[k]['hi']:+.3f}]" for k in RP.GATE))
        cont = set((I35 / "dragon_datasets_test2445/contaminated_ids.txt").read_text().split())
        keep = [i for i, it in enumerate(items) if it["id"] not in cont]
        sub = [items[i] for i in keep]
        a1 = RP.Row("vd", sub, [R["A1 +voting_drop"].sets[i] for i in keep])
        a6 = RP.Row("ours", sub, [R["A6 Ours"].sets[i] for i in keep])
        p = RP.paired(a1, a6, a.n_boot)
        log(f"  without the {len(items) - len(sub)} train-overlapping Test items: Ours - voting_drop " + "  ".join(
            f"{k} {p[k]['gain']:+.3f} [{p[k]['lo']:+.3f}, {p[k]['hi']:+.3f}]" for k in RP.GATE))
    (out / "tables.json").write_text(json.dumps({k: {kk: vv for kk, vv in v.items()} for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
