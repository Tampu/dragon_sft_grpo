#!/usr/bin/env python3
"""Paper analyses 1-5 on Test2445 (InternVL). CPU only (da_env2). Each writes <out-dir>/<name>.txt (+ .json).

  latent  1  latent evidence: greedy recall vs. union recall over K = 1..8 samples, and greedy vs. oracle selection
             over the candidate pool (E + clustered samples) -- on Test and, if present, on the train states
  kcurve  2  accuracy vs. K in {1, 2, 4, 8} for greedy, voting_drop (re-tuned on train at each K) and Ours
             (trained at K = 8, evaluated at smaller K); cost table: generations, output tokens, seconds per item
  cond    3  answer conditioning: GRPO and Ours with the correct / a wrong / no answer; GroupIoU and F1@50 vs gold,
             and the share of predicted boxes that change against the correct-answer prediction
  fail    4  where it fails: keep rate on correct boxes and drop rate on junk by box area; nested duplicates,
             wrong extent on the right element, correct boxes dropped by size
  qual    5  qualitative: one success and one failure figure, 4 panels each (gold / GRPO / pool / Ours), plus the
             ranked candidate lists to pick other examples from
  all        everything whose inputs exist

    /mnt/data2/traviku2/da_env2/bin/python dragon_grpo_r1/paper/analyses.py all --out-dir outputs/paper/test2445/analyses
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
import report_v4 as R4  # noqa: E402

OUT = R1 / "outputs/paper"
AREA_BINS = [(0.0, 0.001, "<0.1%"), (0.001, 0.005, "0.1-0.5%"), (0.005, 0.02, "0.5-2%"), (0.02, 1.01, ">2%")]


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) / 1e6


def macro_mean(vals, doms):
    vals, doms = np.asarray(vals, float), np.asarray(doms)
    return float(np.mean([vals[doms == d].mean() for d in np.unique(doms)]))


def logger(out, name):
    return RP.make_logger(out / f"{name}.txt")


def items_of(path, K=8):
    return [G.completer_item(r, K, prompt="v4t") for r in RP.load_jsonl(path)]


# ------------------------------------------------------------------ 1 latent evidence
def latent(a, out):
    log = logger(out, "latent")
    res = {}
    for split, path in [("Test2445", a.test_dir / "grpo.jsonl"), ("train", a.train_states)]:
        if not Path(path).exists():
            continue
        recs = [r for r in RP.load_jsonl(path) if r["gold"]]          # recall needs gold
        doms = [r["domain"] for r in recs]
        greedy = [len(G.match_pairs(r["E"], r["gold"])) / len(r["gold"]) for r in recs]
        log(f"\n=== {split} (n = {len(recs)} items with gold; macro over domains) ===")
        log(f"{'K':>3} {'greedy R':>9} {'union R':>8} {'oracle-sel GroupIoU':>20} {'oracle-sel F1@50':>17} {'greedy GroupIoU':>16}")
        rows = []
        g_row = RP.Row("greedy", [G.completer_item(r, 1, prompt="v4t") for r in recs], [r["E"] for r in recs])
        for K in range(1, 9):
            union = []
            for r in recs:
                boxes = list(r["E"]) + [b for s in r["samples"][:K] for b in s]
                union.append(sum(any(G.iou(b, g) >= 0.5 for b in boxes) for g in r["gold"]) / len(r["gold"]))
            its = [G.completer_item(r, K, prompt="v4t") for r in recs]
            orc = []
            for it in its:
                cands = it["crops"]
                orc.append([cands[i] for i in G.match_pairs(cands, it["gold"])])
            o = RP.Row(f"oracle K{K}", its, orc)
            row = dict(K=K, greedy_R=macro_mean(greedy, doms), union_R=macro_mean(union, doms),
                       oracle_GroupIoU=o.m["GroupIoU"], oracle_F1=o.m["F1@50"], greedy_GroupIoU=g_row.m["GroupIoU"])
            rows.append(row)
            log(f"{K:3d} {row['greedy_R']:9.3f} {row['union_R']:8.3f} {row['oracle_GroupIoU']:20.3f} "
                f"{row['oracle_F1']:17.3f} {row['greedy_GroupIoU']:16.3f}")
        res[split] = rows
    log("\nunion R: share of gold boxes that some box of E or the first K samples hits at IoU >= 0.5;"
        " oracle-sel: the best keep/drop over the K-sample candidate pool (an upper bound for any selector)")
    (out / "latent.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------ 2 accuracy vs K
def kcurve(a, out):
    log = logger(out, "kcurve")
    path = a.test_dir / "grpo.jsonl"
    E = a.test_dir / "eval"
    tr = RP.load_jsonl(a.train_states) if Path(a.train_states).exists() else None
    log(f"{'K':>3} {'method':>12} {'GroupIoU':>9} {'F1@50':>7} {'gen/item':>9} {'verifier tok':>13}")
    res = []
    for K in [1, 2, 4, 8]:
        its = items_of(path, K)
        g = RP.Row("greedy", its, [i["E"] for i in its])
        rows = [("greedy", g, 1, None)]
        if tr:
            sub = [tr[i] for i in np.random.default_rng(0).choice(len(tr), min(3000, len(tr)), replace=False)]
            f1 = lambda S, gg: (lambda m: 2 * m / (len(S) + len(gg)) if S or gg else 1.0)(len(G.match_pairs(S, gg)))
            kd = G.tune_voting(sub, K, f1)[0]
            rows.append((f"voting {kd[0]}/{kd[1]}", RP.Row("vd", its, [G.voting_drop(i["E"], i["pool"], i["e_sup"], *kd)
                                                                       for i in its]), 1 + K, None))
        sp = E / ("ours" if K == 8 else f"ours_K{K}") / "sel_step400.jsonl"
        if sp.exists():
            sel = RP.load_jsonl(sp)
            rows.append(("Ours", RP.Row("ours", its, R4.v4_sets(its, sel)), 2 + K,
                         float(np.mean([s["n_tokens"] for s in sel]))))
        for name, R, gen, tok in rows:
            res.append(dict(K=K, method=name, GroupIoU=R.m["GroupIoU"], F1=R.m["F1@50"], gen=gen, verifier_tokens=tok))
            log(f"{K:3d} {name:>12} {R.m['GroupIoU']:9.3f} {R.m['F1@50']:7.3f} {gen:9d} "
                f"{'' if tok is None else f'{tok:13.1f}'}")
    recs = RP.load_jsonl(path)
    log(f"\ncost: Grounder greedy output {np.mean([r['E_tokens'] for r in recs]):.1f} tokens/item; samples are drawn "
        f"with the same budget (640). gen/item counts generations (1 greedy + K samples [+ 1 verifier pass]).")
    tp = a.test_dir / "timing.json"
    if tp.exists():
        t = json.load(open(tp))
        log("seconds per item (timing run, one GPU): " + ", ".join(f"{k} {v:.2f}" for k, v in t.items()))
    else:
        log("seconds per item: run ARM=timing in run_ours.sh to measure them")
    (out / "kcurve.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------ 3 answer conditioning
def cond(a, out):
    log = logger(out, "cond")
    T, E = a.test_dir, a.test_dir / "eval"
    base = RP.load_jsonl(T / "grpo.jsonl")
    ids = [r["id"] for r in base]
    its = [G.completer_item(r, 8, prompt="v4t") for r in base]
    preds = {}
    for c, sp, ep in [("correct", T / "grpo.jsonl", E / "ours/sel_step400.jsonl"),
                      ("wrong", T / "grpo_wrong.jsonl", E / "ours_wrong/sel_step400.jsonl"),
                      ("none", T / "grpo_none.jsonl", E / "ours_none/sel_step400.jsonl")]:
        if not sp.exists():
            log(f"  missing {sp}")
            continue
        st = {r["id"]: r for r in RP.load_jsonl(sp)}
        cits = [G.completer_item(st[i], 8, prompt="v4t") for i in ids]
        preds[("GRPO", c)] = [st[i]["E"] for i in ids]
        if ep.exists():
            preds[("Ours", c)] = R4.v4_sets(cits, RP.load_jsonl(ep))
    log(f"{'method':>6} {'answer':>8} {'GroupIoU':>9} {'F1@50':>7} {'boxes changed vs correct':>25}")
    res = []
    for (m, c), S in preds.items():
        R = RP.Row(f"{m} {c}", its, S)
        ref = preds.get((m, "correct"))
        ch = None
        if ref is not None and c != "correct":
            ch = float(np.mean([1 - len(G.match_pairs(s, r)) / max(len(s), len(r)) if (s or r) else 0.0
                                for s, r in zip(S, ref)]))
        res.append(dict(method=m, answer=c, GroupIoU=R.m["GroupIoU"], F1=R.m["F1@50"], changed=ch))
        log(f"{m:>6} {c:>8} {R.m['GroupIoU']:9.3f} {R.m['F1@50']:7.3f} {'' if ch is None else f'{ch:25.3f}'}")
    log("\nboxes changed: per item, 1 - (1:1 matches at IoU 0.5 between this prediction and the correct-answer one) /"
        " the larger set size, averaged. Answer-bearing labels exist only for the train split, so the share of"
        " answer-bearing boxes that change is not computable on Test.")
    wm = Path("/mnt/data2/traviku2/dragon_sft_grpo_internvl35_old/data_paper/test2445_wrong_map.json")
    if wm.exists():
        from collections import Counter
        log("wrong-answer construction: " + str(dict(Counter(v["rule"] for v in json.load(open(wm)).values()))))
    (out / "cond.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------ 4 where it fails
def nested_pairs(boxes):
    """(i, j) pairs where the smaller box lies >= 90% inside the other."""
    out = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            sm, bg = sorted((boxes[i], boxes[j]), key=area)
            ix = max(0, min(sm[2], bg[2]) - max(sm[0], bg[0])) * max(0, min(sm[3], bg[3]) - max(sm[1], bg[1])) / 1e6
            if area(sm) > 0 and ix / area(sm) >= 0.9:
                out.append((i, j))
    return out


def fail(a, out):
    log = logger(out, "fail")
    sp = a.test_dir / "eval/ours/sel_step400.jsonl"
    if not sp.exists():
        log(f"missing {sp}")
        return
    its = items_of(a.test_dir / "grpo.jsonl")
    S = R4.v4_sets(its, RP.load_jsonl(sp))
    log("keep rate on correct candidates / drop rate on junk candidates, by box area (Test2445, Ours)")
    log(f"{'area':>9} {'n hit':>6} {'keep|hit':>9} {'n junk':>7} {'drop|junk':>10}")
    rows = []
    for lo, hi, lab in AREA_BINS:
        kh = nh = dj = nj = 0
        for it, F in zip(its, S):
            pairs = G.match_pairs(it["crops"], it["gold"])
            for k, b in enumerate(it["crops"]):
                if not lo <= area(b) < hi:
                    continue
                kept = any(G.iou(b, f) > 0.99 for f in F)
                if k in pairs:
                    nh += 1; kh += kept
                elif max((G.iou(b, g) for g in it["gold"]), default=0.0) < 0.1:
                    nj += 1; dj += not kept
        rows.append(dict(bin=lab, n_hit=nh, keep_hit=kh / max(1, nh), n_junk=nj, drop_junk=dj / max(1, nj)))
        log(f"{lab:>9} {nh:6d} {kh / max(1, nh):9.3f} {nj:7d} {dj / max(1, nj):10.3f}")
    nested = extent = gold_nested = 0
    dropped = {lab: 0 for _, _, lab in AREA_BINS}
    for it, F in zip(its, S):
        m = G.match_pairs(F, it["gold"])
        nested += sum(1 for i, j in nested_pairs(F) if i not in m or j not in m)
        gold_nested += len(nested_pairs(it["gold"]))
        free = [g for j, g in enumerate(it["gold"]) if j not in m.values()]
        extent += sum(1 for k, f in enumerate(F) if k not in m and any(0.1 <= G.iou(f, g) < 0.5 for g in free))
        for i in G.match_pairs(it["E"], it["gold"]):
            b = it["E"][i]
            if not any(G.iou(b, f) > 0.99 for f in F):
                dropped[next(lab for lo, hi, lab in AREA_BINS if lo <= area(b) < hi)] += 1
    log(f"\nerror counts over {len(its)} items:")
    log(f"  nested duplicates (a final box >= 90% inside another final box, at least one of the two unmatched): {nested}"
        f"  [gold itself has {gold_nested} nested pairs, so nesting alone is not an error]")
    log(f"  wrong extent on the right element (unmatched final box with IoU 0.1-0.5 to an unmatched gold box): {extent}")
    log(f"  Grounder-correct boxes dropped, by area: {dropped}")
    (out / "fail.json").write_text(json.dumps({"bins": rows, "nested": nested, "gold_nested": gold_nested,
                                               "extent": extent, "dropped": dropped},
                                              indent=1))


# ------------------------------------------------------------------ 5 qualitative
def draw(img, boxes, color, width):
    from PIL import ImageDraw
    im = img.copy()
    d = ImageDraw.Draw(im)
    W, H = im.size
    for b in boxes:
        d.rectangle([b[0] * W / 1000, b[1] * H / 1000, b[2] * W / 1000, b[3] * H / 1000], outline=color, width=width)
    return im


def panel_figure(it, grpo, pool, ours, path, title):
    from PIL import Image, ImageDraw
    from gc_engine import VC
    img = VC.load_pil(it["image"]).convert("RGB")
    w = max(2, round(min(img.size) / 250))
    panels = [("gold", draw(img, it["gold"], (0, 160, 0), w)), ("GRPO", draw(img, grpo, (0, 90, 220), w)),
              ("pool", draw(img, pool, (130, 130, 130), max(1, w - 1))), ("Ours", draw(img, ours, (220, 0, 160), w))]
    W, H = img.size
    head = max(36, H // 14)
    fig = Image.new("RGB", (4 * W + 30, H + 2 * head + 10), "white")
    d = ImageDraw.Draw(fig)
    from crops import _font
    font, small = _font(int(head * 0.8)), _font(int(head * 0.5))
    for k, (name, p) in enumerate(panels):
        fig.paste(p, (k * (W + 10), head + 5))
        d.text((k * (W + 10) + 6, 2), name, fill=(0, 0, 0), font=font)
    d.text((6, H + head + 10 + head // 4), title[:120], fill=(90, 90, 90), font=small)
    fig.save(path)


def qual(a, out):
    log = logger(out, "qual")
    sp = a.test_dir / "eval/ours/sel_step400.jsonl"
    if not sp.exists():
        log(f"missing {sp}")
        return
    its = items_of(a.test_dir / "grpo.jsonl")
    S = R4.v4_sets(its, RP.load_jsonl(sp))
    g = RP.Row("g", its, [i["E"] for i in its])
    o = RP.Row("o", its, S)
    gain = o.X[:, 0] - g.X[:, 0]
    succ = [i for i in np.argsort(-gain) if len(its[i]["gold"]) >= 3][:10]
    fails = []
    for i, (it, F) in enumerate(zip(its, S)):
        small = [it["E"][k] for k in G.match_pairs(it["E"], it["gold"]) if area(it["E"][k]) < 0.005
                 and not any(G.iou(it["E"][k], f) > 0.99 for f in F)]
        if small and gain[i] < 0:
            fails.append(i)
    fails = sorted(fails, key=lambda i: gain[i])[:10]
    for kind, lst in [("success", succ), ("failure", fails)]:
        log(f"{kind} candidates (id, GroupIoU GRPO -> Ours):")
        for i in lst:
            log(f"  {its[i]['id']}  {g.X[i, 0]:.3f} -> {o.X[i, 0]:.3f}")
        if lst:
            i = lst[0]
            p = out / f"qual_{kind}.png"
            panel_figure(its[i], its[i]["E"], its[i]["crops"], S[i], p,
                         f"{its[i]['id']}  GroupIoU {g.X[i, 0]:.2f} -> {o.X[i, 0]:.2f}")
            log(f"  figure: {p}")
    log("panel colours: gold green, GRPO blue, pool grey, Ours magenta. To draw another example, call "
        "panel_figure on its id from the lists above.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["latent", "kcurve", "cond", "fail", "qual", "all"])
    ap.add_argument("--test-dir", default=str(OUT / "test2445"))
    ap.add_argument("--train-states", default=str(OUT / "states/train.jsonl"))
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    a.test_dir = Path(a.test_dir)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    fns = {"latent": latent, "kcurve": kcurve, "cond": cond, "fail": fail, "qual": qual}
    for name, fn in fns.items():
        if a.what in (name, "all"):
            print(f"[analyses] {name}", flush=True)
            fn(a, out)


if __name__ == "__main__":
    main()
