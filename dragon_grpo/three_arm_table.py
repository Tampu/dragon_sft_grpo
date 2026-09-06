#!/usr/bin/env python3
"""base vs SFT vs GRPO on the official DRAGON test split, per domain.
Reports MaxIoU, GroupIoU and F1 -- the eval_script.py metrics. CPU only."""
import json, statistics, sys
from pathlib import Path
ROOT = Path("/mnt/data2/traviku2")
DOMS = ["ai2d", "chartqa", "circuitvqa", "infographics", "mapiq", "mapwise"]
NICE = {"ai2d":"AI2D","chartqa":"ChartQA","circuitvqa":"Circuit-VQA",
        "infographics":"InfographicsVQA","mapiq":"MapIQ","mapwise":"MapWise"}
ARMS = [("base","TEST2445_base"),("SFT","TEST2445_sft"),("GRPO","TEST2445_grpo5k_step2500")]

got = {}
for label, d in ARMS:
    p = ROOT/"preds_for_eval"/d/"eval_tau9"/"all_datasets_summary.json"
    if p.exists():
        got[label] = json.load(open(p))["datasets"]
    else:
        print(f"[missing] {label}: {p}")
if not got:
    sys.exit("no arms available")
labels = [l for l, _ in ARMS if l in got]

for key, title in [("IoU","MaxIoU"),("GroupIoU","GroupIoU (GRIT union)"),
                   ("F1@50","F1@50"),("F1@70","F1@70"),("MeanIoU","MeanIoU")]:
    print(f"\n=== {title} ===")
    hdr = f"  {'domain':<17}" + "".join(f"{l:>10}" for l in labels)
    if "base" in got and "GRPO" in got: hdr += f"{'Δ GRPO-base':>13}"
    print(hdr); print("  " + "-"*(17+10*len(labels)+13))
    for dom in DOMS:
        row = f"  {NICE[dom]:<17}" + "".join(f"{got[l][dom][key]:>10.4f}" for l in labels)
        if "base" in got and "GRPO" in got:
            row += f"{got['GRPO'][dom][key]-got['base'][dom][key]:>+13.4f}"
        print(row)
    m = {l: statistics.mean(got[l][d][key] for d in DOMS) for l in labels}
    row = f"  {'MACRO-AVG':<17}" + "".join(f"{m[l]:>10.4f}" for l in labels)
    if "base" in got and "GRPO" in got: row += f"{m['GRPO']-m['base']:>+13.4f}"
    print(row)

print("\n=== empty predictions (no parseable box) ===")
print(f"  {'domain':<17}{'n':>6}" + "".join(f"{l:>10}" for l in labels))
for dom in DOMS:
    cells, n = [], None
    for l in labels:
        s = json.load(open(ROOT/"preds_for_eval"/dict(ARMS)[l]/"eval_tau9"/f"{dom}_summary.json"))
        cells.append(s["num_no_pred"]); n = s["num_samples"]
    print(f"  {NICE[dom]:<17}{n:>6}" + "".join(f"{c:>10}" for c in cells))
