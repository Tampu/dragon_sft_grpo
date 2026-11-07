#!/usr/bin/env python3
"""Rows A-E, per-checkpoint diagnostics and the gates of the pilot. CPU only (da_env2).

  stage1     Val300 rows A-E at each checkpoint, Gate 1 -> gate1.json
  stage2     round-1 system vs an earlier system, Gate 2 / round-stop rule -> gate2.json
  final      Test rows A-E for one seed -> final.json
  check-grounder  exit 1 unless greedy states reproduce a reference run's recall
  bpg-rise   exit 3 if boxes per gold rose more than 30% during a coverage run

Rows go through the unmodified eval_script.py (macro over domains, the way the Test numbers are
reported). The bootstrap uses diagnosis/common.py's per-item versions, and every row prints a warning
if they disagree with eval_script. A malformed completer output falls back to E (edit gain 0) and is
counted in the malformed rate.
"""
from __future__ import annotations

import argparse
import importlib.util as iu
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gc_common as G  # noqa: E402
from gc_engine import ROOT, VC  # noqa: E402

_s = iu.spec_from_file_location("diag_common", ROOT / "diagnosis/common.py")
DC = iu.module_from_spec(_s)
_s.loader.exec_module(DC)

METRICS = ["GroupIoU", "F1@50", "MeanIoU", "F1_1:1", "P_1:1", "R_1:1"]
GATE = ["GroupIoU", "F1@50"]


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def item_vector(gold, pred):
    """[GroupIoU, MeanIoU, m2m R, m2m P, F1_1:1, P_1:1, R_1:1] of one item."""
    r, p, _ = DC.m2m_prf1(gold, pred)
    tp = len(G.hungarian_hits(pred, gold))
    return [DC.group_iou(gold, pred), DC.mean_iou(gold, pred), r, p, DC.bijective_f1(gold, pred),
            tp / len(pred) if len(pred) else 0.0, tp / max(1, len(gold))]


def macro(X, doms):
    """eval_script's aggregation: per-item average per domain, F1 from the averaged R and P, macro."""
    per = []
    for d in np.unique(doms):
        m = X[doms == d].mean(0)
        f1 = 2 * m[2] * m[3] / (m[2] + m[3]) if m[2] + m[3] > 0 else 0.0
        per.append([m[0], f1, m[1], m[4], m[5], m[6]])
    return dict(zip(METRICS, np.mean(per, 0)))


class Row:
    def __init__(self, name, items, sets):
        self.name, self.sets = name, sets
        self.ids = [it["id"] for it in items]
        self.gold = [it["gold"] for it in items]
        self.doms = np.array([it["domain"] for it in items])
        self.X = np.array([item_vector(g, s) for g, s in zip(self.gold, sets)])
        self.m = macro(self.X, self.doms)
        self.bpg = {lab: float(np.mean(v)) if (v := [len(s) / len(g) for g, s in zip(self.gold, sets)
                                                     if lo <= len(g) <= hi]) else float("nan")
                    for lab, lo, hi in G.BINS}
        self.empty = float(np.mean([not s for s in sets]))


def official(row: Row, out_dir: Path):
    recs = [{"id": i, "domain": str(d), "gold": g, "pred": s, "response": "", "n_tokens": 0}
            for i, d, g, s in zip(row.ids, row.doms, row.gold, row.sets)]
    d = out_dir / "preds" / row.name.replace(" ", "_")
    VC.write_pred_dir(d, recs)
    m = VC.run_official_eval(d)["macro"]
    for k in GATE:
        if abs(m[k] - row.m[k]) > 2e-3:
            print(f"[warn] {row.name} {k}: eval_script {m[k]:.4f} vs per-item version {row.m[k]:.4f}")
    row.m.update({"official_" + k: float(m[k]) for k in GATE})
    return row


def paired(a: Row, b: Row, n_boot: int, seed=17):
    """b - a per metric: gain and percentile 95% CI over paired item resamples."""
    if a.ids != b.ids:
        raise SystemExit(f"{a.name} and {b.name} do not cover the same items in the same order")
    rng = np.random.default_rng(seed)
    n = len(a.ids)
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        ma, mb = macro(a.X[i], a.doms[i]), macro(b.X[i], b.doms[i])
        boot.append([mb[k] - ma[k] for k in METRICS])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return {k: dict(gain=float(b.m[k] - a.m[k]), lo=float(lo[j]), hi=float(hi[j])) for j, k in enumerate(METRICS)}


def table(rows, log):
    head = f"{'row':>22} " + " ".join(f"{k:>8}" for k in METRICS) + "   bpg 1-2   3-8    9+  empty"
    log(head)
    for r in rows:
        log(f"{r.name:>22} " + " ".join(f"{r.m[k]:8.3f}" for k in METRICS) +
            f"  {r.bpg['1-2']:8.2f} {r.bpg['3-8']:5.2f} {r.bpg['9+']:5.2f} {r.empty:6.3f}")


def fmt_paired(p):
    return "  ".join(f"{k} {p[k]['gain']:+.3f} [{p[k]['lo']:+.3f},{p[k]['hi']:+.3f}]" for k in METRICS[:4])


def final_sets(items, sel):
    """Selected set per item, malformed -> E."""
    by = {s["id"]: s for s in sel}
    return [G.selected(it["E"], it["pool"], by[it["id"]]["idx"]) if by[it["id"]]["idx"] is not None else list(it["E"])
            for it in items]


def diagnostics(items, sel, n_boot=2000):
    by = {s["id"]: s for s in sel}
    A = {k: [] for k in ("gain", "n_add", "add_hit", "n_drop", "drop_junk", "listed", "keep", "pool_n", "pool_hit",
                         "tokens")}
    sup, base_sup = Counter(), Counter()
    bad = trunc = 0
    for it in items:
        s = by[it["id"]]
        E, pool, gold = it["E"], it["pool"], it["gold"]
        flags = G.pool_hit_flags(E, pool, gold)
        for c, f in zip(pool, flags):
            base_sup[(c["support"], f)] += 1
        idx = s["idx"]
        bad += idx is None
        trunc += bool(s["trunc"])
        st = G.edit_stats(E, pool, gold, idx) if idx is not None else \
            dict(n_add=0, add_hit=0, n_drop=0, drop_junk=0, n_keep=len(E), by_support=Counter())
        sup.update(st["by_support"])
        gain = G.phi(G.selected(E, pool, idx), gold) - G.phi(E, gold) if idx is not None else 0.0
        for k, v in (("gain", gain), ("n_add", st["n_add"]), ("add_hit", st["add_hit"]), ("n_drop", st["n_drop"]),
                     ("drop_junk", st["drop_junk"]), ("listed", len(E) + len(pool)),
                     ("keep", st["n_keep"] + st["n_add"]), ("pool_n", len(pool)), ("pool_hit", sum(flags)),
                     ("tokens", s["n_tokens"])):
            A[k].append(v)
    A = {k: np.array(v, float) for k, v in A.items()}
    rate = lambda a, b: float(a.sum() / b.sum()) if b.sum() else float("nan")
    d = dict(n=len(items), edit_gain=float(A["gain"].mean()), adds=float(A["n_add"].mean()),
             drops=float(A["n_drop"].mean()), add_prec=rate(A["add_hit"], A["n_add"]),
             drop_prec=rate(A["drop_junk"], A["n_drop"]), pool_base=rate(A["pool_hit"], A["pool_n"]),
             pool_size=float(A["pool_n"].mean()), keep_rate=rate(A["keep"], A["listed"]),
             malformed=bad / len(items), truncated=trunc / len(items), mean_tokens=float(A["tokens"].mean()),
             pool_recall=float(np.mean([G.pool_recall(i["E"], i["pool"], i["gold"]) for i in items])))
    rng = np.random.default_rng(17)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(items), len(items))
        if A["n_add"][i].sum() and A["pool_n"][i].sum():
            diffs.append(rate(A["add_hit"][i], A["n_add"][i]) - rate(A["pool_hit"][i], A["pool_n"][i]))
    d["add_prec_minus_base_ci"] = [float(x) for x in np.percentile(diffs, [2.5, 97.5])] if diffs else [float("nan")] * 2
    d["by_support"] = {}
    for k in sorted({a for a, _ in list(sup) + list(base_sup)}):
        na, nb = sup[(k, True)] + sup[(k, False)], base_sup[(k, True)] + base_sup[(k, False)]
        d["by_support"][str(k)] = dict(adds=na, add_prec=sup[(k, True)] / na if na else None,
                                       pool=nb, base=base_sup[(k, True)] / nb if nb else None)
    return d


def fmt_diag(d):
    bs = "  ".join(f"s={k}: adds {v['adds']} prec {v['add_prec'] if v['add_prec'] is None else round(v['add_prec'], 3)}"
                   f" vs base {v['base'] if v['base'] is None else round(v['base'], 3)}"
                   for k, v in d["by_support"].items())
    return (f"edit gain {d['edit_gain']:+.4f}  adds {d['adds']:.2f}  drops {d['drops']:.2f}  add prec {d['add_prec']:.3f}"
            f" (pool base {d['pool_base']:.3f}, diff CI [{d['add_prec_minus_base_ci'][0]:+.3f},"
            f"{d['add_prec_minus_base_ci'][1]:+.3f}])  drop prec {d['drop_prec']:.3f}  pool recall {d['pool_recall']:.3f}"
            f"  pool size {d['pool_size']:.1f}  keep rate {d['keep_rate']:.3f}  malformed {d['malformed']:.3f}"
            f"  truncated {d['truncated']:.3f}  tokens {d['mean_tokens']:.1f}\n      by support: {bs}")


def choose_vote_k(train_recs, K, log):
    """k by mean of GroupIoU and F1@50 on the TRAIN states."""
    items = [G.completer_item(r, K) for r in train_recs]
    best, score = None, -1.0
    for k in range(1, K + 1):
        m = Row(f"vote{k}", items, [G.voting(i["E"], i["pool"], k) for i in items]).m
        s = (m["GroupIoU"] + m["F1@50"]) / 2
        log(f"  train voting k={k}: GroupIoU {m['GroupIoU']:.3f} F1@50 {m['F1@50']:.3f}")
        if s > score:
            best, score = k, s
    log(f"  chosen k={best}")
    return best, items


def state_summary(items):
    return dict(phi_E=float(np.mean([G.phi(i["E"], i["gold"]) for i in items])),
                recall_E=float(np.mean([len(G.hungarian_hits(i["E"], i["gold"])) / len(i["gold"]) for i in items])),
                pool_recall=float(np.mean([G.pool_recall(i["E"], i["pool"], i["gold"]) for i in items])),
                pool_size=float(np.mean([len(i["pool"]) for i in items])))


def make_logger(path):
    f = open(path, "w")

    def log(s=""):
        print(s, flush=True)
        f.write(s + "\n"); f.flush()
    return log


def cmd_stage1(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    log = make_logger(out / "report.txt")
    val = load_jsonl(a.val_states)
    items = [G.completer_item(r, a.K) for r in val]
    k_vote, train_items = choose_vote_k(load_jsonl(a.train_states), a.K, log)
    log(f"[states] train {state_summary(train_items)}")
    log(f"[states] val   {state_summary(items)}")
    K_all = len(val[0]["samples"])
    recall_by_K = {K: float(np.mean([G.pool_recall(r["E"], G.build_pool(r["E"], r["samples"][:K]), r["gold"]) for r in val]))
                   for K in sorted({a.K, K_all}) if K}
    log(f"[pool recall on val by K] {recall_by_K}")

    sft = {r["id"]: r["E"] for r in load_jsonl(a.sft_states)}
    A = official(Row("A SFT greedy", items, [sft[i["id"]] for i in items]), out)
    B = official(Row("B grounder greedy", items, [i["E"] for i in items]), out)
    C = official(Row(f"C voting k={k_vote}", items, [G.voting(i["E"], i["pool"], k_vote) for i in items]), out)
    E = official(Row("E oracle edit", items, [G.oracle_edit(i["E"], i["pool"], i["gold"]) for i in items]), out)

    steps, best = [int(s) for s in a.steps.split(",")], None
    D_rows = {}
    for st in steps:
        sel = load_jsonl(Path(a.sel_dir) / f"sel_step{st}.jsonl")
        D = official(Row(f"D completer step{st}", items, final_sets(items, sel)), out)
        vsB, vsC = paired(B, D, a.n_boot), paired(C, D, a.n_boot)
        diag = diagnostics(items, sel)
        D_rows[st] = dict(row=D, vsB=vsB, vsC=vsC, diag=diag)
        score = np.mean([vsB[k]["gain"] for k in GATE])
        if best is None or score > best[0]:
            best = (score, st)
    st = best[1]
    rows = [A, B, C] + [D_rows[s]["row"] for s in steps] + [E]
    log()
    table(rows, log)
    log()
    for s in steps:
        r = D_rows[s]
        log(f"step{s}  vs B: {fmt_paired(r['vsB'])}\n        vs C: {fmt_paired(r['vsC'])}\n      {fmt_diag(r['diag'])}")
        if r["diag"]["malformed"] > 0.05:
            log(f"[warn] step{s}: {r['diag']['malformed']:.3f} malformed; they count as E, so read D with that in mind")
    b = D_rows[st]
    checks = {
        "D-B >= +0.02 on GroupIoU, CI above 0": b["vsB"]["GroupIoU"]["gain"] >= a.min_gain and b["vsB"]["GroupIoU"]["lo"] > 0,
        "D-B >= +0.02 on F1@50, CI above 0": b["vsB"]["F1@50"]["gain"] >= a.min_gain and b["vsB"]["F1@50"]["lo"] > 0,
        "D > C on GroupIoU": b["vsC"]["GroupIoU"]["gain"] > 0,
        "D > C on F1@50": b["vsC"]["F1@50"]["gain"] > 0,
        "add precision above pool base (paired CI above 0)": b["diag"]["add_prec_minus_base_ci"][0] > 0,
        "bpg on 1-2 gold not up > 10% vs B": b["row"].bpg["1-2"] <= 1.10 * B.bpg["1-2"],
    }
    go = all(checks.values())
    log(f"\n=== GATE 1 at best step {st} (chosen on Val by mean D-B gain) ===")
    for k, v in checks.items():
        log(f"  {'PASS' if v else 'FAIL'}  {k}")
    log(f"  -> {'GO to Stage 2' if go else 'STOP: do not run Stage 2'}")
    (out / "gate1.json").write_text(json.dumps({
        "go": bool(go), "best_step": st, "adapter": str(Path(a.ckpt_dir) / f"grpo-step{st}"),
        "sel": str(Path(a.sel_dir) / f"sel_step{st}.jsonl"), "val_states": a.val_states, "K": a.K, "k_vote": k_vote,
        "checks": {k: bool(v) for k, v in checks.items()}, "vsB": b["vsB"], "vsC": b["vsC"], "diag": b["diag"],
        "rows": {r.name: r.m for r in rows}, "recall_by_K": {str(k): v for k, v in recall_by_K.items()}}, indent=1))


def cmd_stage2(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    log = make_logger(out / "report.txt")
    prev_items = [G.completer_item(r, a.K) for r in load_jsonl(a.prev_states)]
    prev_sel = load_jsonl(a.prev_sel)
    S = official(Row("previous system", prev_items, final_sets(prev_items, prev_sel)), out)
    B = official(Row("B previous grounder", prev_items, [i["E"] for i in prev_items]), out)
    prev_diag = diagnostics(prev_items, prev_sel)
    C = official(Row(f"C voting k={a.k_vote} prev", prev_items, [G.voting(i["E"], i["pool"], a.k_vote) for i in prev_items]), out)
    items = [G.completer_item(r, a.K) for r in load_jsonl(a.new_states)]
    # both systems' rows must line up item by item for the paired bootstrap
    order = {i["id"]: n for n, i in enumerate(prev_items)}
    items.sort(key=lambda i: order[i["id"]])
    B1 = official(Row("new grounder greedy", items, [i["E"] for i in items]), out)
    C1 = official(Row(f"C voting k={a.k_vote} new", items, [G.voting(i["E"], i["pool"], a.k_vote) for i in items]), out)
    E1 = official(Row("new oracle edit", items, [G.oracle_edit(i["E"], i["pool"], i["gold"]) for i in items]), out)
    res, best = {}, None
    for st in [int(s) for s in a.steps.split(",")]:
        sel = load_jsonl(Path(a.sel_dir) / f"sel_step{st}.jsonl")
        R = official(Row(f"round system step{st}", items, final_sets(items, sel)), out)
        res[st] = dict(row=R, vs=paired(S, R, a.n_boot), diag=diagnostics(items, sel))
        score = np.mean([res[st]["vs"][k]["gain"] for k in GATE])
        if best is None or score > best[0]:
            best = (score, st)
    st = best[1]
    table([B, C, S, B1, C1] + [r["row"] for r in res.values()] + [E1], log)
    log(f"\nprevious system: {fmt_diag(prev_diag)}")
    for s, r in res.items():
        log(f"step{s} vs previous: {fmt_paired(r['vs'])}\n      {fmt_diag(r['diag'])}")
    r = res[st]
    d = r["diag"]
    prev_prec, prec = prev_diag["pool_base"], d["pool_base"]
    collusion = {
        "pool size up > 50% while pool precision falls": d["pool_size"] > 1.5 * prev_diag["pool_size"] and prec < prev_prec,
        "keep rate falling (by more than 0.02)": d["keep_rate"] < prev_diag["keep_rate"] - 0.02,
        "bpg up on 1-2 gold (by more than 10%)": r["row"].bpg["1-2"] > 1.10 * S.bpg["1-2"],
    }
    checks = {f"beats previous by >= {a.min_gain} on {k}, CI above 0": r["vs"][k]["gain"] >= a.min_gain and r["vs"][k]["lo"] > 0
              for k in GATE}
    checks["pool recall up"] = d["pool_recall"] > prev_diag["pool_recall"]
    checks.update({f"no collusion: {k}": not v for k, v in collusion.items()})
    go = all(checks.values())
    improved_any = any(r["vs"][k]["gain"] >= a.min_gain and r["vs"][k]["lo"] > 0 for k in GATE)
    log(f"\n=== GATE at best step {st} ===")
    for k, v in checks.items():
        log(f"  {'PASS' if v else 'FAIL'}  {k}")
    log(f"  -> {'GO' if go else 'NO-GO'}")
    (out / "gate2.json").write_text(json.dumps({
        "go": bool(go), "improved_any": bool(improved_any), "collusion": any(collusion.values()), "best_step": st,
        "adapter": str(Path(a.ckpt_dir) / f"grpo-step{st}"), "sel": str(Path(a.sel_dir) / f"sel_step{st}.jsonl"),
        "val_states": a.new_states, "K": a.K, "checks": {k: bool(v) for k, v in checks.items()}, "vs": r["vs"],
        "diag": d, "prev_diag": prev_diag, "rows": {x["row"].name: x["row"].m for x in res.values()}}, indent=1))


def cmd_v3(a):
    """v3 completer (EDGE prompt, reading order, no support, crop sheet) vs the round-0 v1 completer on the SAME
    states. v3 wins only if its best Val step (1) beats round 0 by >= min_gain on GroupIoU and F1@50 with CI above
    0, (2) beats voting_drop (tuned on TRAIN, 1:1 F1) with CI above 0 on both, (3) is malformed on < 2% of items.
    With --ablate-sel, also reports how much each ablation costs the best step (are the crops used?)."""
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    log = make_logger(out / "report.txt")
    items = [G.completer_item(r, a.K) for r in load_jsonl(a.states)]
    tr = load_jsonl(a.train_states)
    if len(tr) > 3000:
        tr = [tr[i] for i in np.random.default_rng(0).choice(len(tr), 3000, replace=False)]
    f1 = lambda S, g: (lambda m: 2 * m / (len(S) + len(g)) if S or g else 1.0)(len(G.match_pairs(S, g)))
    kd, _, _ = G.tune_voting(tr, a.K, f1)
    log(f"[voting_drop] tuned on {len(tr)} train states (1:1 F1): k_add={kd[0]} k_keep={kd[1]}")
    B = official(Row("B grounder greedy", items, [i["E"] for i in items]), out)
    VD = official(Row(f"voting_drop {kd[0]}/{kd[1]}", items,
                      [G.voting_drop(i["E"], i["pool"], i["e_sup"], *kd) for i in items]), out)
    r0_sel = load_jsonl(a.r0_sel)
    R0 = official(Row("round-0 completer v1", items, final_sets(items, r0_sel)), out)
    res, best = {}, None
    for st in [int(s) for s in a.steps.split(",")]:
        sel = load_jsonl(Path(a.sel_dir) / f"sel_step{st}.jsonl")
        R = official(Row(f"v3 step{st}", items, final_sets(items, sel)), out)
        res[st] = dict(row=R, vs_r0=paired(R0, R, a.n_boot), vs_vd=paired(VD, R, a.n_boot), diag=diagnostics(items, sel),
                       malformed=float(np.mean([s["idx"] is None for s in sel])))
        score = np.mean([res[st]["vs_r0"][k]["gain"] for k in GATE])
        if best is None or score > best[0]:
            best = (score, st)
    st = best[1]
    r = res[st]
    abl = {}
    for spec in a.ablate_sel or []:
        name, path = spec.split("=", 1)
        A = official(Row(f"v3 step{st} {name}", items, final_sets(items, load_jsonl(path))), out)
        abl[name] = dict(row=A, cost=paired(A, r["row"], a.n_boot))
    table([B, VD, R0] + [x["row"] for x in res.values()] + [x["row"] for x in abl.values()], log)
    log(f"\nround-0 completer: {fmt_diag(diagnostics(items, r0_sel))}")
    for s, x in res.items():
        log(f"v3 step{s} vs round 0: {fmt_paired(x['vs_r0'])}\n      vs voting_drop: {fmt_paired(x['vs_vd'])}"
            f"\n      malformed {x['malformed']:.3f}  {fmt_diag(x['diag'])}")
    for name, x in abl.items():
        log(f"best step minus {name}: {fmt_paired(x['cost'])}")
    checks = {}
    for k in GATE:
        checks[f"beats round 0 by >= {a.min_gain} on {k}, CI above 0"] = \
            r["vs_r0"][k]["gain"] >= a.min_gain and r["vs_r0"][k]["lo"] > 0
    for k in GATE:
        checks[f"beats voting_drop on {k}, CI above 0"] = r["vs_vd"][k]["lo"] > 0
    checks["malformed < 2%"] = r["malformed"] < 0.02
    go = all(checks.values())
    log(f"\n=== v3 DECISION at best step {st} ===")
    for k, v in checks.items():
        log(f"  {'PASS' if v else 'FAIL'}  {k}")
    log(f"  -> {'WIN' if go else 'NO WIN'}")
    (out / "v3_decision.json").write_text(json.dumps({
        "win": bool(go), "best_step": st, "adapter": str(Path(a.ckpt_dir) / f"grpo-step{st}"),
        "sel": str(Path(a.sel_dir) / f"sel_step{st}.jsonl"), "voting_drop_k": list(kd),
        "checks": {k: bool(v) for k, v in checks.items()}, "vs_r0": r["vs_r0"], "vs_voting_drop": r["vs_vd"],
        "ablations": {n: x["cost"] for n, x in abl.items()}, "malformed": r["malformed"],
        "rows": {x.name: x.m for x in [B, VD, R0] + [y["row"] for y in res.values()] + [y["row"] for y in abl.values()]}},
        indent=1, default=float))


def cmd_compare_e(a):
    """Greedy E of two state files on the same items (e.g. the grounder under its v1 vs v3 system prompt)."""
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    log = make_logger(out / "report.txt")
    ra, rb = load_jsonl(a.a), {r["id"]: r for r in load_jsonl(a.b)}
    ra = [r for r in ra if r["id"] in rb]
    A = official(Row(a.name_a, ra, [r["E"] for r in ra]), out)
    B = official(Row(a.name_b, ra, [rb[r["id"]]["E"] for r in ra]), out)
    table([A, B], log)
    vs = paired(A, B, a.n_boot)
    log(f"{a.name_b} minus {a.name_a}: {fmt_paired(vs)}")
    trunc = lambda rs: float(np.mean([r.get("E_trunc", False) for r in rs]))
    log(f"truncated: {a.name_a} {trunc(ra):.3f}  {a.name_b} {trunc(list(rb.values())):.3f}")
    (out / "compare_e.json").write_text(json.dumps({"vs": vs, "rows": {A.name: A.m, B.name: B.m}}, indent=1, default=float))


def cmd_coevo(a):
    """Keep co-evolution only if the alpha=0.5 round (1) beats round 0 by >= min_gain on GroupIoU and F1@50 with
    CI above 0, (2) beats the alpha=0 round on both, and (3) its Val pool recall is not below round 0's.
    (1) and (3) come from the coupled run's own round report; (2) is a fresh paired comparison."""
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    log = make_logger(out / "decision.txt")
    g5 = json.load(open(Path(a.coupled) / "r1/report/gate2.json"))
    g0 = json.load(open(Path(a.control) / "r1/report/gate2.json"))
    ns = argparse.Namespace(prev_states=g0["val_states"], prev_sel=g0["sel"], new_states=g5["val_states"],
                            sel_dir=str(Path(g5["sel"]).parent), ckpt_dir=str(Path(g5["adapter"]).parent),
                            steps=str(g5["best_step"]), K=a.K, min_gain=0.0, k_vote=a.k_vote, n_boot=a.n_boot,
                            out_dir=str(out / "coupled_vs_control"))
    cmd_stage2(ns)
    vs0 = json.load(open(out / "coupled_vs_control" / "gate2.json"))["vs"]
    ci = lambda v: f"{v['gain']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]"
    checks = {}
    for k in GATE:
        checks[f"alpha 0.5 beats round 0 by >= {a.min_gain} on {k} (CI above 0): {ci(g5['vs'][k])}"] = \
            g5["vs"][k]["gain"] >= a.min_gain and g5["vs"][k]["lo"] > 0
    for k in GATE:
        checks[f"alpha 0.5 beats alpha 0 on {k}: {ci(vs0[k])}"] = vs0[k]["gain"] > 0
    pr5, pr0, prc = g5["diag"]["pool_recall"], g5["prev_diag"]["pool_recall"], g0["diag"]["pool_recall"]
    checks[f"pool does not shrink: Val pool recall {pr5:.3f} vs round 0 {pr0:.3f} (alpha 0: {prc:.3f})"] = pr5 >= pr0
    log("\n=== CO-EVOLUTION DECISION ===")
    for k, v in checks.items():
        log(f"  {'PASS' if v else 'FAIL'}  {k}")
    keep = all(checks.values())
    log(f"  -> {'KEEP the co-evolution claim' if keep else 'DROP the co-evolution claim: the method is the round-0 completer'}")
    log(f"  for reading only: alpha 0.5 vs alpha 0 must be above 0 in point estimate; the CI is shown, not required")
    (out / "decision.json").write_text(json.dumps({"keep": keep, "checks": {k: bool(v) for k, v in checks.items()},
                                                   "coupled_vs_round0": g5["vs"], "coupled_vs_control": vs0,
                                                   "pool_recall": {"coupled": pr5, "round0": pr0, "control": prc}}, indent=1))


def cmd_final(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    log = make_logger(out / "report.txt")
    items = [G.completer_item(r, a.K) for r in load_jsonl(a.system_states)]
    order = {it["id"]: n for n, it in enumerate(items)}
    k_vote, _ = choose_vote_k(load_jsonl(a.train_states), a.K, log)
    frozen = {r["id"]: r["E"] for r in load_jsonl(a.frozen_states)}
    sft = {r["id"]: r["E"] for r in load_jsonl(a.sft_states)}
    if set(order) != set(frozen):
        raise SystemExit("system and frozen-grounder states cover different items")
    sel = load_jsonl(a.sel)
    rows = [official(Row(n, items, s), out) for n, s in (
        ("A SFT greedy", [sft[i["id"]] for i in items]),
        ("B frozen grounder", [frozen[i["id"]] for i in items]),
        (f"C voting k={k_vote}", [G.voting(i["E"], i["pool"], k_vote) for i in items]),
        ("D system", final_sets(items, sel)),
        ("E oracle edit", [G.oracle_edit(i["E"], i["pool"], i["gold"]) for i in items]))]
    table(rows, log)
    log(f"\nD vs B: {fmt_paired(paired(rows[1], rows[3], a.n_boot))}\nD vs C: {fmt_paired(paired(rows[2], rows[3], a.n_boot))}")
    diag = diagnostics(items, sel)
    log(fmt_diag(diag))
    (out / "final.json").write_text(json.dumps({"rows": {r.name: r.m for r in rows}, "diag": diag,
                                                "generations_per_item": 1 + a.K + 1}, indent=1))


def cmd_check_grounder(a):
    xy = lambda b: [b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]]
    ref = {x["sample_id"]: [xy(b) for b in x["pred_boxes_parsed"]] for x in json.load(open(a.ref_pred))}
    recs = load_jsonl(a.states)
    rec = lambda f: float(np.mean([len(G.hungarian_hits(f(r), r["gold"])) / len(r["gold"]) for r in recs]))
    got, want = rec(lambda r: r["E"]), rec(lambda r: ref[r["id"]])
    same = np.mean([[list(map(float, b)) for b in r["E"]] == ref[r["id"]] for r in recs])
    print(f"[grounder check] recall {got:.4f} vs reference {want:.4f}; identical outputs {same:.3f}")
    sys.exit(1 if abs(got - want) > a.tol else 0)


def cmd_bpg_rise(a):
    v = [r["bpg"] for r in load_jsonl(a.metrics) if r.get("bpg") is not None]
    w = min(a.window, max(1, len(v) // 2))
    first, last = float(np.mean(v[:w])), float(np.mean(v[-w:]))
    print(f"[bpg] first {w} steps {first:.2f}, last {w} steps {last:.2f}, ratio {last / first:.2f}")
    sys.exit(3 if last > 1.3 * first else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stage1")
    for n in ("val-states", "train-states", "sft-states", "sel-dir", "ckpt-dir", "out-dir", "steps"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--min-gain", type=float, default=0.02)
    p.add_argument("--n-boot", type=int, default=5000)
    p.set_defaults(fn=cmd_stage1)
    p = sub.add_parser("stage2")
    for n in ("prev-states", "prev-sel", "new-states", "sel-dir", "ckpt-dir", "out-dir", "steps"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--min-gain", type=float, default=0.01)
    p.add_argument("--k-vote", type=int, default=2)
    p.add_argument("--n-boot", type=int, default=5000)
    p.set_defaults(fn=cmd_stage2)
    p = sub.add_parser("v3")
    for n in ("states", "train-states", "r0-sel", "sel-dir", "ckpt-dir", "out-dir", "steps"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--ablate-sel", action="append", metavar="NAME=PATH")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--min-gain", type=float, default=0.01)
    p.add_argument("--n-boot", type=int, default=5000)
    p.set_defaults(fn=cmd_v3)
    p = sub.add_parser("compare-e")
    for n in ("a", "b", "out-dir"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--name-a", default="grounder v1 prompt")
    p.add_argument("--name-b", default="grounder v3 prompt")
    p.add_argument("--n-boot", type=int, default=5000)
    p.set_defaults(fn=cmd_compare_e)
    p = sub.add_parser("coevo")
    p.add_argument("--coupled", required=True, help="seed dir of the alpha=0.5 run, e.g. outputs/gc_pilot/s3v2_a05/seed17")
    p.add_argument("--control", required=True, help="seed dir of the alpha=0 run")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--k-vote", type=int, default=3)
    p.add_argument("--min-gain", type=float, default=0.01)
    p.add_argument("--n-boot", type=int, default=5000)
    p.set_defaults(fn=cmd_coevo)
    p = sub.add_parser("final")
    for n in ("system-states", "frozen-states", "sft-states", "train-states", "sel", "out-dir"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--n-boot", type=int, default=5000)
    p.set_defaults(fn=cmd_final)
    p = sub.add_parser("check-grounder")
    p.add_argument("--states", required=True)
    p.add_argument("--ref-pred", required=True)
    p.add_argument("--tol", type=float, default=0.01)
    p.set_defaults(fn=cmd_check_grounder)
    p = sub.add_parser("bpg-rise")
    p.add_argument("--metrics", required=True)
    p.add_argument("--window", type=int, default=30)
    p.set_defaults(fn=cmd_bpg_rise)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
