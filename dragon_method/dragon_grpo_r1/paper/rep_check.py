#!/usr/bin/env python3
"""Pool representative check, done before the fresh Grounder states exist. CPU only, no retraining.

Uses the OLD Grounder's states (train states for tuning, Val600 for scoring). Clustering and pool membership
are exactly build_pool's; only the box that stands for each cluster changes:

  current   E boxes as greedy wrote them; pool clusters by medoid   (= today; also the "greedy E variant" choice,
            since a pool cluster never has an E member: build_pool drops clusters E already covers)
  medoid    each E box -> medoid of {E box + the best-matching box of every sample with IoU >= 0.5}; pool medoid
  median    same groups, coordinate-wise median, for E boxes and pool clusters
  median_pool   E boxes kept; pool clusters by median (separates the pool effect from the E effect)

Per variant: voting_drop (k_add, k_keep re-tuned on train states, 1:1 F1, as report_paper does) and the
oracle edit (E hits + one pool cluster per unmatched gold), scored with the official GroupIoU / F1@50 macro and
1:1 F1, with paired 95% CIs against current. Rule (fixed in advance): adopt a variant only if its voting_drop
F1@50 gain is >= 0.01 with CI above 0; otherwise keep current.

  /mnt/data2/traviku2/da_env2/bin/python dragon_grpo_r1/paper/rep_check.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

R1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R1 / "completer_pilot"))
import gc_common as G  # noqa: E402
import report as RP  # noqa: E402

GC = R1 / "outputs/gc_pilot"
VARIANTS = ["current", "medoid", "median", "median_pool"]


def median(members):
    return [float(x) for x in np.median(np.array(members, float), 0)]


def pool_clusters(E, samples, iou_merge=0.5, iou_covered=0.5, max_pool=30):
    """build_pool, but returning the members too (same clusters, same filter, same order, same cap)."""
    cl = []
    for si, s in enumerate(samples):
        for b in G._dedupe(s, iou_merge):
            for c in cl:
                if G.iou(b, c["rep"]) >= iou_merge:
                    c["members"].append(b)
                    c["sids"].add(si)
                    break
            else:
                cl.append({"rep": b, "members": [b], "sids": {si}})
    pool = []
    for c in cl:
        rep = G._medoid(c["members"])
        if all(G.iou(rep, e) < iou_covered for e in E):
            pool.append({"rep": list(rep), "members": c["members"], "support": len(c["sids"]), "sids": sorted(c["sids"])})
    pool.sort(key=lambda c: -c["support"])
    return pool[:max_pool]


def e_groups(E, samples, thr=0.5):
    """Per E box: itself + each sample's best box with IoU >= thr to it."""
    out = []
    for e in E:
        g = [list(e)]
        for s in samples:
            best = max(s, key=lambda b: G.iou(e, b), default=None)
            if best is not None and G.iou(e, best) >= thr:
                g.append(list(best))
        out.append(g)
    return out


def variant(rec, v, K=8):
    E = [list(b) for b in rec["E"]]
    smp = rec["samples"][:K]
    pool = pool_clusters(E, smp)
    es = G.e_support(E, smp)
    if v in ("medoid", "median"):
        f = (lambda m: list(G._medoid(m))) if v == "medoid" else median
        E = [f(g) for g in e_groups(E, smp)]
    if v in ("median", "median_pool"):
        for c in pool:
            c["rep"] = median(c["members"])
    assert pool == [] or all("members" in c for c in pool)
    return E, pool, es


def f1_11(S, g):
    m = len(G.match_pairs(S, g))
    return 2 * m / (len(S) + len(g)) if (S or g) else 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=str(GC / "s3/seed17/r0/train.jsonl"))
    ap.add_argument("--val", default=str(GC / "coevo/v3c_vB/seed17/r1/val600/g_prev.jsonl"))
    ap.add_argument("--n-tune", type=int, default=3000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", default=str(R1 / "outputs/paper/rep_check"))
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = RP.make_logger(out / "report.txt")
    tr = RP.load_jsonl(a.train)
    tr = [tr[i] for i in np.random.default_rng(0).choice(len(tr), min(a.n_tune, len(tr)), replace=False)]
    val = RP.load_jsonl(a.val)
    items = [{"id": r["id"], "domain": r["domain"], "gold": [list(b) for b in r["gold"]]} for r in val]
    log(f"tune: {len(tr)} old-Grounder train states   score: {len(val)} Val600 states ({a.val})")

    rows, res = {}, {}
    for v in VARIANTS:
        prep = [(*variant(r, v), r["gold"]) for r in tr]
        best = max(((ka, kk) for ka in range(1, 9) for kk in range(0, 9)),
                   key=lambda k: np.mean([f1_11(G.voting_drop(E, p, es, *k), g) for E, p, es, g in prep]))
        V = [variant(r, v) for r in val]
        sets = {"voting_drop": [G.voting_drop(E, p, es, *best) for E, p, es in V],
                "oracle": [G.oracle_edit(E, p, it["gold"]) for (E, p, es), it in zip(V, items)],
                "E only": [E for E, p, es in V]}
        for kind, S in sets.items():
            rows[(v, kind)] = RP.Row(f"{v}/{kind}", items, S)
        res[v] = {"k": best}
        log(f"  {v:12s} voting_drop tuned k_add={best[0]} k_keep={best[1]}")

    log(f"\n{'variant':12s} {'row':12s} {'GroupIoU':>8s} {'F1@50':>7s} {'F1 1:1':>7s}   vs current: "
        f"{'dGroupIoU [95% CI]':>24s} {'dF1@50 [95% CI]':>24s}")
    adopt = []
    for kind in ["voting_drop", "oracle", "E only"]:
        for v in VARIANTS:
            r = rows[(v, kind)]
            s = f"{v:12s} {kind:12s} {r.m['GroupIoU']:8.4f} {r.m['F1@50']:7.4f} {r.m['F1_1:1']:7.4f}"
            if v != "current":
                p = RP.paired(rows[("current", kind)], r, a.n_boot)
                s += "   " + " ".join(f"{p[k]['gain']:+.4f} [{p[k]['lo']:+.4f},{p[k]['hi']:+.4f}]" for k in RP.GATE)
                res[v][kind] = {k: p[k] for k in RP.GATE + ["F1_1:1"]}
                if kind == "voting_drop" and p["F1@50"]["gain"] >= 0.01 and p["F1@50"]["lo"] > 0:
                    adopt.append(v)
            log(s)
        log("")
    pick = max(adopt, key=lambda v: res[v]["voting_drop"]["F1@50"]["gain"]) if adopt else "current"
    log(f"DECISION (voting_drop F1@50 gain >= 0.01, CI > 0): {pick}"
        + ("" if adopt else "  -> keep the current representative"))
    res["decision"] = pick
    (out / "rep_check.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
