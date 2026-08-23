#!/usr/bin/env python3
"""
Build the 2000-sample raw pool for the real (non-overfit) v4 SFT run, mixed
across all 6 DRAGON datasets.

Two filters applied per-sample (on the normalized [0,1000] xyxy gt boxes,
same normalize_and_sort_gt_boxes() the v4 converter itself uses -- so a
sample that survives here is guaranteed to survive cmd_convert too):

  1. drop if > MAX_BOXES boxes -- these are annotation-noise-heavy targets
     (dense box lists the earlier Phase 0 verify showed the model cannot
     reproduce; keeping them in the SFT pool would teach the model to hedge
     towards generic grid-like box layouts instead of precise localization).
  2. drop if ANY box covers >= FULL_IMAGE_FRAC of the image in both width
     and height -- a "whole image" box is not localized evidence, it's
     annotation noise (e.g. mapwise's occasional (0,0,1000,999) box mixed
     into an otherwise-real box list) and actively hurts format learning.

Domain balance: rather than sampling proportional to each dataset's raw
size (which would let chartqa/ai2d dominate purely because their raw
samples.jsonl files happen to be larger), each of the 6 datasets gets an
equal target quota of TOTAL // 6, with remainder distributed to the first
datasets alphabetically. If a dataset's post-filter pool can't fill its
quota (mapiq's raw pool is only 300, smaller than the other 5's 500), the
shortfall is redistributed round-robin across datasets that still have
surplus, so the total still lands at (as close as possible to) TOTAL
without silently over-representing whichever datasets are largest.

A third filter, discovered when sft_v4_phase0.py's `convert` step refused
to write at this scale: some (image, question) pairs (mostly mapiq, a
couple chartqa) appear twice with two DIFFERENT gold box sets -- annotator
disagreement or a duplicate-id bug upstream. These are unfittable by
definition (the model can't learn one correct answer for an input that has
two contradictory targets), so every record sharing a contradictory
(image, question) key is dropped here, before quota allocation -- not
patched over with --allow-contradictions at convert time.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from sft_v3_patches import normalize_and_sort_gt_boxes
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = ROOT / "Diagram_Attribution_Dataset"
DRAGON_DIR = ROOT / "dragon_datasets"
OUT_DIR = ROOT / "dragon_datasets_sft2k"

SEED = 42
TOTAL = 2000
MAX_BOXES = 15
FULL_IMAGE_FRAC = 0.90  # box covering >=90% of width AND height == noise

DATASET_KEYS = [
    "ai2d_grounding",
    "chartqa_grounding",
    "circuitvqa_grounding",
    "infographics_grounding",
    "mapiq_grounding",
    "mapwise_grounding",
]


def is_full_image_box(box, frac=FULL_IMAGE_FRAC):
    x1, y1, x2, y2 = box
    return (x2 - x1) >= frac * 1000 and (y2 - y1) >= frac * 1000


def load_valid_pool(key: str):
    """Return (valid_records, drop_counts dict) for one dataset."""
    samples_path = DRAGON_DIR / key / "samples.jsonl"
    records = [json.loads(l) for l in samples_path.read_text().splitlines() if l.strip()]

    candidates = []  # (rec, boxes) that pass the per-record filters
    drops = {"missing_image": 0, "no_boxes": 0, "too_many_boxes": 0, "full_image": 0}
    for rec in records:
        image_rel = rec.get("image_path") or rec.get("image")
        image_abs = (IMAGE_ROOT / image_rel).resolve()
        if not image_abs.exists():
            drops["missing_image"] += 1
            continue
        with Image.open(image_abs) as img:
            image_size = img.size

        boxes = normalize_and_sort_gt_boxes(rec.get("bbox") or [], image_size)
        if not boxes:
            drops["no_boxes"] += 1
            continue
        if len(boxes) > MAX_BOXES:
            drops["too_many_boxes"] += 1
            continue
        if any(is_full_image_box(b) for b in boxes):
            drops["full_image"] += 1
            continue

        candidates.append((rec, tuple(boxes)))

    # contradiction check: same (image, question) -> different gold box sets
    # (same key convention as sft_v4_phase0.py's cmd_convert)
    key_to_boxsets: dict[tuple, set] = {}
    for rec, boxes in candidates:
        image_rel = rec.get("image_path") or rec.get("image")
        k = (str(image_rel), (rec.get("question") or "").strip().lower())
        key_to_boxsets.setdefault(k, set()).add(boxes)
    contradictory_keys = {k for k, boxsets in key_to_boxsets.items() if len(boxsets) > 1}

    drops["contradictory"] = 0
    valid = []
    for rec, boxes in candidates:
        image_rel = rec.get("image_path") or rec.get("image")
        k = (str(image_rel), (rec.get("question") or "").strip().lower())
        if k in contradictory_keys:
            drops["contradictory"] += 1
            continue
        valid.append(rec)
    return valid, drops


def allocate_quotas(pool_sizes: dict[str, int], total: int) -> dict[str, int]:
    """Equal split with remainder to the first keys, then redistribute any
    shortfall (a dataset whose pool is smaller than its quota) round-robin
    across datasets that still have surplus capacity."""
    keys = list(pool_sizes.keys())
    n = len(keys)
    base = total // n
    remainder = total - base * n
    quota = {k: base + (1 if i < remainder else 0) for i, k in enumerate(keys)}

    # clamp to pool size, track shortfall
    shortfall = 0
    for k in keys:
        if quota[k] > pool_sizes[k]:
            shortfall += quota[k] - pool_sizes[k]
            quota[k] = pool_sizes[k]

    # redistribute shortfall round-robin to datasets with remaining surplus
    while shortfall > 0:
        capacity = {k: pool_sizes[k] - quota[k] for k in keys if pool_sizes[k] - quota[k] > 0}
        if not capacity:
            break  # no dataset can absorb more; total will land under `total`
        share = max(1, shortfall // len(capacity))
        progressed = False
        for k in capacity:
            if shortfall <= 0:
                break
            add = min(share, capacity[k], shortfall)
            if add > 0:
                quota[k] += add
                shortfall -= add
                progressed = True
        if not progressed:
            break
    return quota


def main() -> None:
    rng = random.Random(SEED)  # single rng reused sequentially, matches this repo's convention

    pools, all_drops = {}, {}
    for key in DATASET_KEYS:
        valid, drops = load_valid_pool(key)
        pools[key] = valid
        all_drops[key] = drops
        raw_n = len(valid) + sum(drops.values())
        print(f"[{key}] raw={raw_n}  valid={len(valid)}  "
              f"dropped(missing_image={drops['missing_image']}, "
              f"no_boxes={drops['no_boxes']}, "
              f">15boxes={drops['too_many_boxes']}, "
              f"full_image={drops['full_image']}, "
              f"contradictory={drops['contradictory']})")

    pool_sizes = {k: len(v) for k, v in pools.items()}
    quota = allocate_quotas(pool_sizes, TOTAL)

    selected = []
    print("\nDomain allocation (quota / valid pool):")
    for key in DATASET_KEYS:
        records = pools[key]
        rng.shuffle(records)
        n = quota[key]
        chosen = records[:n]
        selected.extend(chosen)
        print(f"  {key}: {len(chosen)} / {pool_sizes[key]}")

    print(f"\nTotal selected: {len(selected)} (target {TOTAL})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "raw_sft2k_records.jsonl"
    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in selected) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(selected)} raw records to {out_path}")


if __name__ == "__main__":
    main()
