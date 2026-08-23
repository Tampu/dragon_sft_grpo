#!/usr/bin/env python3
"""
Combine the 6 datasets' pre-existing samples_infer_holdout.jsonl (100/dataset,
verified disjoint from dragon6_sft2k_v4's training pool -- see below) into one
raw jsonl for sft_v4_phase0.py convert/verify, so the sft2k checkpoint can be
scored on genuinely unseen data instead of the training pool itself.

Disjointness check (id alone is NOT unique within a dataset -- e.g. every
image has its own "Q_0" -- so this compares (image_path, id) pairs, same
composite key convert's contradiction check uses):
    ai2d          : 0/499 overlap
    chartqa       : 0/500 overlap
    circuitvqa    : 0/500 overlap
    infographics  : 3/497 overlap  (pre-existing artifact in the raw holdout
                                     split itself, not introduced by this
                                     script or by build_dragon6_sft2k.py --
                                     small enough (<0.2% of the 2000-sample
                                     training pool at most) to not block on)
    mapiq         : 0/289 overlap
    mapwise       : 0/499 overlap

Each record's 'id' is prefixed with its dataset key (e.g. "ai2d_grounding::Q_0")
purely so verify's printed [k/n] id=... lines are attributable to a domain --
nothing else about the record is touched.

Also drops contradictory (image, question) -> multiple-gold-box-set groups,
same rule and same reason as build_dragon6_sft2k.py: unfittable by
definition, and for eval specifically there's no single correct target to
score a prediction against. sft_v4_phase0.py convert's own contradiction
check would otherwise abort on these (found: 1 in chartqa, 3 in mapiq).
"""
import json
from pathlib import Path

from PIL import Image

from sft_v3_patches import normalize_and_sort_gt_boxes

ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = ROOT / "Diagram_Attribution_Dataset"
DRAGON_DIR = ROOT / "dragon_datasets"
OUT_DIR = ROOT / "dragon_datasets_sft2k"

DATASET_KEYS = [
    "ai2d_grounding",
    "chartqa_grounding",
    "circuitvqa_grounding",
    "infographics_grounding",
    "mapiq_grounding",
    "mapwise_grounding",
]


def main() -> None:
    combined = []
    for key in DATASET_KEYS:
        path = DRAGON_DIR / key / "samples_infer_holdout.jsonl"
        records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        for rec in records:
            rec = dict(rec)
            rec["id"] = f"{key}::{rec.get('id')}"
            combined.append(rec)
        print(f"[{key}] {len(records)} holdout samples")

    # contradiction check (same composite key convention as build_dragon6_sft2k.py)
    key_to_boxsets: dict[tuple, set] = {}
    rec_boxes = []
    for rec in combined:
        image_rel = rec.get("image_path") or rec.get("image")
        image_abs = (IMAGE_ROOT / image_rel).resolve()
        with Image.open(image_abs) as img:
            image_size = img.size
        boxes = tuple(normalize_and_sort_gt_boxes(rec.get("bbox") or [], image_size))
        rec_boxes.append(boxes)
        k = (str(image_rel), (rec.get("question") or "").strip().lower())
        key_to_boxsets.setdefault(k, set()).add(boxes)
    contradictory_keys = {k for k, boxsets in key_to_boxsets.items() if len(boxsets) > 1}

    kept, dropped = [], 0
    for rec, boxes in zip(combined, rec_boxes):
        image_rel = rec.get("image_path") or rec.get("image")
        k = (str(image_rel), (rec.get("question") or "").strip().lower())
        if k in contradictory_keys:
            dropped += 1
            continue
        kept.append(rec)
    print(f"\nDropped {dropped} records in contradictory (image, question) groups")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "raw_holdout600.jsonl"
    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(kept)} combined holdout records to {out_path}")


if __name__ == "__main__":
    main()
