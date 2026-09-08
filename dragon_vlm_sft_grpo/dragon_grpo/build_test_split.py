#!/usr/bin/env python3
"""
Convert the six per-domain Test JSONLs (build_dragon6_raw.py's output,
already uniformly filtered -- no bbox / missing image dropped, nothing
else) into this repo's chat-conversation schema, one file per domain, ready
for export_for_eval.py.

Also writes a contaminated_ids.txt listing which test questions also appear
in the SFT or GRPO training pools -- an UPSTREAM defect in the released
split_reviewed-2 splits (Train/Val/Test are not fully disjoint), not
something this pipeline introduces; reported so a contamination-free
variant can be scored alongside the full-set number, the same audit
dragon_sft_grpo's build_test2445.py performs for the InternVL pipeline.

Usage:
    python3 build_test_split.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from box_geometry import normalize_and_sort_gt_boxes  # noqa: E402
from grounding_prompts import GROUNDING_SYSTEM_PROMPT, build_target, build_user_prompt  # noqa: E402

RAW_TEST_DIR = ROOT / "dragon_datasets" / "raw_by_split" / "Test"
IMAGE_ROOT = ROOT / "Diagram_Attribution_Dataset"
OUT_DIR = ROOT / "dragon_datasets" / "test_split"

DOMAIN_KEYS = ["ai2d", "chartqa", "circuitvqa", "infographics", "mapiq", "mapwise"]


def load_training_keys() -> set:
    keys = set()
    for name in ("raw_sft_records.jsonl", "raw_grpo_records.jsonl"):
        path = ROOT / "dragon_datasets" / name
        if not path.exists():
            continue
        for l in path.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                keys.add((r["image_path"], str(r["id"])))
    return keys


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    training_keys = load_training_keys()

    contaminated, total = [], 0
    for key in DOMAIN_KEYS:
        raw_path = RAW_TEST_DIR / f"{key}.jsonl"
        raw_records = [json.loads(l) for l in raw_path.read_text().splitlines() if l.strip()]

        converted = []
        for rec in raw_records:
            image_rel = rec["image_path"]
            image_abs = (IMAGE_ROOT / image_rel).resolve()
            if not image_abs.exists():
                continue
            from PIL import Image
            with Image.open(image_abs) as img:
                image_size = img.size
            boxes = normalize_and_sort_gt_boxes(rec.get("bbox") or [], image_size)
            if not boxes:
                continue

            if (image_rel, str(rec["id"])) in training_keys:
                contaminated.append(rec["id"])

            converted.append({
                "id": rec["id"],
                "image": image_rel,
                "conversations": [
                    {"from": "system", "value": GROUNDING_SYSTEM_PROMPT},
                    {"from": "user", "value": build_user_prompt(
                        question=rec.get("question", ""),
                        choices=rec.get("choices") or [],
                        answer=rec.get("answer", ""))},
                    {"from": "assistant", "value": build_target(boxes)},
                ],
                "metadata": {"answer": rec.get("answer", ""), "image_size": list(image_size),
                             "gt_boxes_norm": [list(b) for b in boxes], "domain": key},
            })

        out_path = OUT_DIR / f"{key}.jsonl"
        out_path.write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in converted) + "\n",
            encoding="utf-8",
        )
        total += len(converted)
        print(f"  [{key}] {len(converted)} usable test samples -> {out_path}")

    (OUT_DIR / "contaminated_ids.txt").write_text("\n".join(contaminated) + "\n")
    print(f"\ntotal usable: {total}")
    print(f"test questions also present in SFT/GRPO training pools: {len(contaminated)} "
          f"(upstream split-overlap defect -- see DATASET.md) -> {OUT_DIR/'contaminated_ids.txt'}")


if __name__ == "__main__":
    main()
