#!/usr/bin/env python3
"""
thin_box_analysis.py -- the cause -> mechanism -> fix evidence pipeline for
the scale-adaptive reward term (--w-scale in grpo_dragon6_v1.py).

The contribution claim is NOT "we added a distance term". It is:
  CAUSE     diagram evidence is often thin (axis lines, text rows, legend
            strips); recall@0.9 is pinned near zero SPECIFICALLY on thin
            boxes, across checkpoints (empirical, Sec. A below);
  MECHANISM for a box of min-side s offset by d along its thin axis,
            IoU = (s-d)/(s+d), so IoU >= 0.9 requires d <= s/19 -- for an
            8-unit text row that is ~0.4/1000 units, i.e. sub-pixel; IoU's
            gradient is also exactly zero once overlap is lost (analytic,
            Sec. B below);
  FIX       a reward term whose tolerance adapts to the target's own scale
            (0.5*IoU + 0.5*exp(-center_dist/gt_diagonal)) restores gradient
            for thin targets; the test is that training with it lifts
            recall@0.9 IN THE THIN BINS while chunky bins stay flat
            (comparative, Sec. C below).

Usage -- Sections A+B (no GPU, run NOW on existing predictions):

    python thin_box_analysis.py \
        --preds sft=preds_for_eval/sft2k/pred_mapiq.json \
                grpo=preds_for_eval/grpo_step250/pred_mapiq.json \
        --out thin_analysis_mapiq.json

    Accepts any number of label=path pairs; files are eval_script.py-schema
    pred_<dataset>.json. Pass several domains' files under one label by
    repeating the label: --preds sft=a.json sft=b.json grpo=c.json ...

Section C is the same command run again after the w_scale ablation exists,
adding e.g.  grpo_scale=preds_for_eval/grpo_scale_pilot/pred_mapiq.json --
the printed per-bin DELTA table is the paper figure.

RUN PLAN for the ablation (so the fix is attributable):
  1. [now, CPU]     Sections A+B on SFT + GRPO-step250 preds -> failure proven.
  2. [pilot, ~2.5h] 500-prompt GRPO, SAME pool/seed as the original pilot,
                    --w-scale 0.15 --w-iou 0.30 (localization budget kept
                    constant), abstention fix + winning no_repeat_ngram in
                    both arms. The ONLY delta vs the original pilot is the
                    scale term.
  3. [analysis]     Section C: recall@0.9 per thinness bin, pilot vs
                    scale-pilot. Success = lift concentrated in bins <=20,
                    chunky bins ~unchanged, val f1 not degraded.
  4. [decision]     If (3) positive, the term goes in the paper as the
                    derived contribution WITH this figure; whether it also
                    goes into the main 5k run is separate -- keep the 5k
                    on the base reward so the headline number and the
                    ablation stay independently attributable.
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# thinness = min(w, h) of the GT box in the file's coordinate units
# (our exports are [0,1000]-normalized, so bins are in 0-1000 units)
BINS = [(0, 10, "hairline <=10"), (10, 20, "thin 10-20"), (20, 50, "small 20-50"),
        (50, 120, "medium 50-120"), (120, 10**9, "chunky >120")]
TAUS = [0.5, 0.7, 0.9]


def iou_xywh(a: Dict, b: Dict) -> float:
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / ua if ua > 0 else 0.0


def bin_of(min_side: float) -> str:
    for lo, hi, name in BINS:
        if lo <= min_side < hi:
            return name
    return BINS[-1][2]


def analyze_file(path: Path) -> Dict[str, Dict]:
    """Per thinness bin: n GT boxes, mean best IoU, recall@tau."""
    items = json.loads(path.read_text())
    acc = defaultdict(lambda: {"n": 0, "best_ious": []})
    for it in items:
        gt = it.get("gt_boxes_raw", [])
        pred = it.get("pred_boxes_parsed", [])
        for g in gt:
            min_side = min(g["w"], g["h"])
            b = bin_of(min_side)
            best = max((iou_xywh(g, p) for p in pred), default=0.0)
            acc[b]["n"] += 1
            acc[b]["best_ious"].append(best)
    out = {}
    for name, d in acc.items():
        ious = d["best_ious"]
        out[name] = {"n_gt_boxes": d["n"],
                     "mean_best_iou": round(statistics.mean(ious), 4) if ious else 0.0}
        for t in TAUS:
            out[name][f"recall@{t}"] = round(
                sum(1 for v in ious if v >= t) / len(ious), 4) if ious else 0.0
    return out


def merge_label(paths: List[Path]) -> Dict[str, Dict]:
    """Combine several domain files under one label by pooling GT boxes."""
    pooled = defaultdict(lambda: {"n": 0, "best_ious": []})
    for p in paths:
        items = json.loads(p.read_text())
        for it in items:
            gt = it.get("gt_boxes_raw", [])
            pred = it.get("pred_boxes_parsed", [])
            for g in gt:
                b = bin_of(min(g["w"], g["h"]))
                best = max((iou_xywh(g, p_) for p_ in pred), default=0.0)
                pooled[b]["n"] += 1
                pooled[b]["best_ious"].append(best)
    out = {}
    for name, d in pooled.items():
        ious = d["best_ious"]
        out[name] = {"n_gt_boxes": d["n"],
                     "mean_best_iou": round(statistics.mean(ious), 4) if ious else 0.0}
        for t in TAUS:
            out[name][f"recall@{t}"] = round(
                sum(1 for v in ious if v >= t) / len(ious), 4) if ious else 0.0
    return out


def analytic_table() -> List[Dict]:
    """SECTION B -- the mechanism, no data needed. For a prediction identical
    in size to the GT but offset d along the thin axis (min side s):
        IoU(d) = (s - d) / (s + d)
    so IoU >= tau  <=>  d <= s * (1 - tau) / (1 + tau).
    At tau=0.9 that's d <= s/19."""
    rows = []
    for s in [6, 8, 10, 15, 25, 60, 150, 300]:
        row = {"min_side": s}
        for t in TAUS:
            row[f"max_offset@{t}"] = round(s * (1 - t) / (1 + t), 2)
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", nargs="+", required=True,
                    help="label=path pairs, e.g. sft=pred_mapiq.json grpo=pred2.json; "
                         "repeat a label to pool multiple domain files under it")
    ap.add_argument("--out", default="thin_box_analysis.json")
    args = ap.parse_args()

    by_label: Dict[str, List[Path]] = defaultdict(list)
    for spec in args.preds:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--preds entries must be label=path, got: {spec}")
        by_label[label].append(Path(path))

    results = {label: merge_label(paths) for label, paths in by_label.items()}
    labels = list(results.keys())
    bin_names = [name for _, _, name in BINS]

    # ---------------- SECTION A: the empirical failure ----------------
    print("=" * 78)
    print("A. EMPIRICAL: recall@tau stratified by GT-box thinness (min side, /1000)")
    print("=" * 78)
    for label in labels:
        print(f"\n[{label}]")
        print(f"  {'bin':<16} {'n_boxes':>8} {'meanIoU':>8} " +
              " ".join(f"{'R@'+str(t):>7}" for t in TAUS))
        for bn in bin_names:
            r = results[label].get(bn)
            if not r:
                continue
            print(f"  {bn:<16} {r['n_gt_boxes']:>8} {r['mean_best_iou']:>8.3f} " +
                  " ".join(f"{r[f'recall@{t}']:>7.3f}" for t in TAUS))
    print("\n  Failure signature to report: R@0.9 near zero in hairline/thin bins")
    print("  for EVERY checkpoint, while chunky bins reach it -- the floor is a")
    print("  property of the metric x geometry, not of any one model.")

    # ---------------- SECTION B: the analytic mechanism ----------------
    print("\n" + "=" * 78)
    print("B. ANALYTIC: max placement offset (units/1000) that still clears IoU=tau")
    print("   (prediction same size as GT, offset along thin axis: IoU=(s-d)/(s+d))")
    print("=" * 78)
    print(f"  {'min_side':>9} " + " ".join(f"{'d@'+str(t):>8}" for t in TAUS))
    for row in analytic_table():
        print(f"  {row['min_side']:>9} " +
              " ".join(f"{row[f'max_offset@{t}']:>8.2f}" for t in TAUS))
    print("\n  Read: an 8-unit text row tolerates 0.42 units of offset at tau=0.9 --")
    print("  sub-pixel on typical diagram resolutions -- and IoU's gradient is")
    print("  exactly zero once overlap is lost, so RL gets no signal to close a")
    print("  near-miss on thin evidence. This is the derivation of the")
    print("  scale-adaptive term (tolerance normalized by the target's own scale).")

    # ---------------- SECTION C: the fix (needs >=2 labels) ----------------
    if len(labels) >= 2:
        base, others = labels[0], labels[1:]
        print("\n" + "=" * 78)
        print(f"C. COMPARATIVE: per-bin DELTA vs '{base}' -- the fix is validated iff")
        print("   the R@0.9 lift concentrates in hairline/thin bins with chunky ~flat")
        print("=" * 78)
        for other in others:
            print(f"\n[{other} - {base}]")
            print(f"  {'bin':<16} {'dMeanIoU':>9} " +
                  " ".join(f"{'dR@'+str(t):>8}" for t in TAUS))
            for bn in bin_names:
                a, b = results[base].get(bn), results[other].get(bn)
                if not a or not b:
                    continue
                print(f"  {bn:<16} {b['mean_best_iou']-a['mean_best_iou']:>+9.3f} " +
                      " ".join(f"{b[f'recall@{t}']-a[f'recall@{t}']:>+8.3f}" for t in TAUS))

    Path(args.out).write_text(json.dumps(
        {"bins": [b[2] for b in BINS], "taus": TAUS,
         "analytic": analytic_table(), "results": results}, indent=2))
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
