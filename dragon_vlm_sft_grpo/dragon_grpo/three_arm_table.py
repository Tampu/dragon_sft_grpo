#!/usr/bin/env python3
"""base vs SFT vs GRPO on the test split, per domain. Reports MaxIoU,
GroupIoU, F1, MeanIoU -- the eval_script.py metrics. CPU only.

Usage:
    python3 three_arm_table.py --base preds_for_eval/base \
        --sft preds_for_eval/sft --grpo preds_for_eval/grpo
Each dir is expected to already contain an eval/all_datasets_summary.json
(i.e. run_eval_arms.sh's eval_script.py step has been run for that arm).
"""
import argparse
import json
import statistics
from pathlib import Path

DOMS = ["ai2d", "chartqa", "circuitvqa", "infographics", "mapiq", "mapwise"]
NICE = {"ai2d": "AI2D", "chartqa": "ChartQA", "circuitvqa": "Circuit-VQA",
        "infographics": "InfographicsVQA", "mapiq": "MapIQ", "mapwise": "MapWise"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None)
    ap.add_argument("--sft", default=None)
    ap.add_argument("--grpo", default=None)
    args = ap.parse_args()

    arms = [("base", args.base), ("SFT", args.sft), ("GRPO", args.grpo)]
    got = {}
    for label, d in arms:
        if not d:
            continue
        p = Path(d) / "eval" / "all_datasets_summary.json"
        if p.exists():
            got[label] = json.loads(p.read_text())["datasets"]
        else:
            print(f"[missing] {label}: {p}")
    if not got:
        raise SystemExit("no arms available")
    labels = [l for l, _ in arms if l in got]

    for key, title in [("IoU", "MaxIoU"), ("GroupIoU", "GroupIoU (GRIT union)"),
                       ("F1@50", "F1@50"), ("F1@70", "F1@70"), ("MeanIoU", "MeanIoU")]:
        print(f"\n=== {title} ===")
        hdr = f"  {'domain':<17}" + "".join(f"{l:>10}" for l in labels)
        if "base" in got and "GRPO" in got:
            hdr += f"{'Δ GRPO-base':>13}"
        print(hdr); print("  " + "-" * (17 + 10 * len(labels) + 13))
        for dom in DOMS:
            row = f"  {NICE[dom]:<17}" + "".join(f"{got[l][dom][key]:>10.4f}" for l in labels)
            if "base" in got and "GRPO" in got:
                row += f"{got['GRPO'][dom][key] - got['base'][dom][key]:>+13.4f}"
            print(row)
        m = {l: statistics.mean(got[l][d][key] for d in DOMS) for l in labels}
        row = f"  {'MACRO-AVG':<17}" + "".join(f"{m[l]:>10.4f}" for l in labels)
        if "base" in got and "GRPO" in got:
            row += f"{m['GRPO'] - m['base']:>+13.4f}"
        print(row)

    print("\n=== empty predictions (no parseable box) ===")
    print(f"  {'domain':<17}{'n':>6}" + "".join(f"{l:>10}" for l in labels))
    arm_dirs = dict(arms)
    for dom in DOMS:
        cells, n = [], None
        for l in labels:
            d = arm_dirs["base" if l == "base" else ("sft" if l == "SFT" else "grpo")]
            s = json.loads((Path(d) / "eval" / f"{dom}_summary.json").read_text())
            cells.append(s["num_no_pred"]); n = s["num_samples"]
        print(f"  {NICE[dom]:<17}{n:>6}" + "".join(f"{c:>10}" for c in cells))


if __name__ == "__main__":
    main()
