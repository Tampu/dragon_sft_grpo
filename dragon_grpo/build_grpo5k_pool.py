#!/usr/bin/env python3
"""
Build the ~5000-sample GRPO training pool, guaranteed disjoint from both the
2000-sample SFT training pool (dragon_datasets/<key>/samples.jsonl) and the
592-sample SFT holdout-eval pool (dragon_datasets/<key>/samples_infer_holdout.jsonl).

Why this needs a fresh walk instead of just "take what's left in samples.jsonl":
samples.jsonl and samples_infer_holdout.jsonl were themselves only a 500/300
(train) + 100 (val) PER-DATASET SUBSET of split_reviewed-2's full raw pool
(prepare_dragon6_exp2.py) -- Train alone has 1122-1400 raw annotated JSONs per
dataset (see split_reviewed-2/Train/<folder>), of which only 500 (300 for
mapiq) were ever converted. There's much more real, never-touched data
available than "2800 minus 2000" would suggest -- except for MapIQ, which is
a genuine ceiling: only 336/1122 of its raw Train files have boxes at all
(see prepare_dragon6_exp2.py's docstring), and 300 of those 336 are already
in samples.jsonl. So MapIQ contributes far less than an equal 1/6 share here,
by hard data availability, not by choice -- same shape of constraint as the
SFT pool's mapiq shortfall, just tighter.

Filters (deliberately DIFFERENT from build_dragon6_sft2k.py's 3 filters --
see the box-count discussion below):
  1. drop if any box covers >=90% of the image (full-image noise) -- kept,
     same reasoning as SFT: not localized evidence.
  2. drop contradictory (image, question) -> multiple-gold-box-set groups --
     kept, unfittable by definition regardless of training paradigm.
  3. the SFT pool's ">15 boxes" filter is DELIBERATELY NOT applied here.
     grpo_dragon6_v1.py's reward function is explicitly designed around
     dense multi-box targets (its w_miss term's docstring literally
     discusses "on a 15-box target going 2->3 boxes" as the case it exists
     to give gradient on) -- excluding exactly the samples that motivate
     that reward term would defeat the point of RL over SFT here.

Sourcing: walks split_reviewed-2/{Train,Val}/<folder> per dataset with the
same seed=42 + shuffle convention prepare_dragon6_exp2.py used, skips any
candidate already present in samples.jsonl or samples_infer_holdout.jsonl
(checked by (image_path, id) -- id alone repeats across images), keeps
everything else that has boxes and survives the two filters above, then
applies the same equal-quota-with-shortfall-redistribution balancing as
build_dragon6_sft2k.py, target total 5000.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from PIL import Image
from sft_v3_patches import normalize_and_sort_gt_boxes  # noqa: E402

SPLIT_DIR = ROOT / "split_reviewed-2"
IMAGE_ROOT = ROOT / "Diagram_Attribution_Dataset"
DRAGON_DIR = ROOT / "dragon_datasets"
OUT_DIR = ROOT / "dragon_datasets_grpo"

SEED = 42
TOTAL = 5000
FULL_IMAGE_FRAC = 0.90

# key -> split_reviewed-2 folder name (same mapping as prepare_dragon6_exp2.py)
DATASET_FOLDERS = {
    "ai2d_grounding": "ai2d",
    "chartqa_grounding": "ChartQA",
    "circuitvqa_grounding": "Circuit-VQA",
    "infographics_grounding": "Infographics",
    "mapiq_grounding": "MapIQ",
    "mapwise_grounding": "Mapwise",
}

ANSWER_KEYS = ["gt_answer", "ground_truth_answer"]


def clean_choices(raw):
    if not isinstance(raw, list) or not raw:
        return []
    cleaned = [str(x).strip() for x in raw if str(x).strip()]
    if any(any(ch in c for ch in "{}':") for c in cleaned):
        return []
    return cleaned


def has_boxes(raw: dict) -> bool:
    bbox = raw.get("bbox")
    if not isinstance(bbox, list) or not bbox:
        bbox = raw.get("boxes")
    return isinstance(bbox, list) and len(bbox) > 0


def to_adapter_record(raw: dict, q_id_fallback: str) -> dict:
    """Same shape as prepare_dragon6_exp2.py's to_adapter_record -- must
    match, since this record needs to look identical to a samples.jsonl row
    for sft_v4_phase0.py convert to consume it."""
    if "answers" in raw and isinstance(raw["answers"], dict):
        answer = raw["answers"].get("correct", "")
        explanation_raw = raw["answers"].get("predicted_explanation", "")
    else:
        answer = ""
        for key in ANSWER_KEYS:
            if raw.get(key):
                answer = raw[key]
                break
        explanation_raw = raw.get("predicted_explanation", "")

    bbox = raw.get("bbox")
    if not isinstance(bbox, list) or not bbox:
        bbox = raw.get("boxes", [])

    return {
        "id": raw.get("q_id", q_id_fallback),
        "question": raw.get("question_text", ""),
        "choices": clean_choices(raw.get("choices")),
        "answer": str(answer).strip(),
        "image_path": raw["image_path"],
        "bbox": bbox,
        "explanation_raw": explanation_raw,
    }


def is_full_image_box(box, frac=FULL_IMAGE_FRAC):
    x1, y1, x2, y2 = box
    return (x2 - x1) >= frac * 1000 and (y2 - y1) >= frac * 1000


def load_used_keys(key: str) -> set:
    """(image_path, id) pairs already spent on SFT-train or SFT-eval."""
    used = set()
    for fname in ("samples.jsonl", "samples_infer_holdout.jsonl"):
        path = DRAGON_DIR / key / fname
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            image_rel = rec.get("image_path") or rec.get("image")
            used.add((str(image_rel), str(rec.get("id"))))
    return used


def walk_split(key: str, folder_name: str, split: str, used_keys: set):
    """Yield valid, never-used adapter-contract records from one split_reviewed-2
    Train/ or Val/ folder, in the same seed=42-shuffled order exp2 used."""
    src_folder = SPLIT_DIR / split / folder_name
    if not src_folder.exists():
        return
    candidates = sorted(src_folder.glob("*.json"))
    rng = random.Random(SEED)
    rng.shuffle(candidates)

    for i, fp in enumerate(candidates):
        raw = json.loads(fp.read_text())
        if not has_boxes(raw):
            continue
        record = to_adapter_record(raw, q_id_fallback=f"{folder_name}-{split}-{i}")
        image_rel = record["image_path"]
        rec_key = (str(image_rel), str(record["id"]))
        if rec_key in used_keys:
            continue  # already spent on SFT-train or SFT-eval

        image_abs = (IMAGE_ROOT / image_rel).resolve()
        if not image_abs.exists():
            continue
        with Image.open(image_abs) as img:
            image_size = img.size
        boxes = normalize_and_sort_gt_boxes(record.get("bbox") or [], image_size)
        if not boxes:
            continue
        if any(is_full_image_box(b) for b in boxes):
            continue

        yield record, tuple(boxes)


def allocate_quotas(pool_sizes: dict, total: int) -> dict:
    """Identical strategy to build_dragon6_sft2k.py's allocate_quotas."""
    keys = list(pool_sizes.keys())
    n = len(keys)
    base = total // n
    remainder = total - base * n
    quota = {k: base + (1 if i < remainder else 0) for i, k in enumerate(keys)}

    shortfall = 0
    for k in keys:
        if quota[k] > pool_sizes[k]:
            shortfall += quota[k] - pool_sizes[k]
            quota[k] = pool_sizes[k]

    while shortfall > 0:
        capacity = {k: pool_sizes[k] - quota[k] for k in keys if pool_sizes[k] - quota[k] > 0}
        if not capacity:
            break
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
    rng = random.Random(SEED)  # reused sequentially across datasets for the final quota draw

    pools = {}
    for key, folder_name in DATASET_FOLDERS.items():
        used_keys = load_used_keys(key)
        candidates = []
        for split in ("Train", "Val"):
            candidates.extend(walk_split(key, folder_name, split, used_keys))

        # contradiction check within this dataset's fresh pool
        key_to_boxsets: dict = {}
        for record, boxes in candidates:
            image_rel = record["image_path"]
            k = (str(image_rel), (record.get("question") or "").strip().lower())
            key_to_boxsets.setdefault(k, set()).add(boxes)
        contradictory_keys = {k for k, bs in key_to_boxsets.items() if len(bs) > 1}

        valid = []
        n_contradictory = 0
        for record, boxes in candidates:
            image_rel = record["image_path"]
            k = (str(image_rel), (record.get("question") or "").strip().lower())
            if k in contradictory_keys:
                n_contradictory += 1
                continue
            valid.append(record)

        pools[key] = valid
        print(f"[{key}] fresh usable pool (unused by SFT, has boxes, no full-image, "
              f"no contradiction): {len(valid)}  (dropped {n_contradictory} contradictory)")

    pool_sizes = {k: len(v) for k, v in pools.items()}
    total_available = sum(pool_sizes.values())
    quota = allocate_quotas(pool_sizes, TOTAL)

    selected = []
    print("\nDomain allocation (quota / fresh valid pool):")
    for key in DATASET_FOLDERS:
        records = pools[key]
        rng.shuffle(records)
        n = quota[key]
        chosen = records[:n]
        selected.extend(chosen)
        print(f"  {key}: {len(chosen)} / {pool_sizes[key]}")

    print(f"\nTotal selected: {len(selected)} (target {TOTAL}, total fresh available {total_available})")
    if len(selected) < TOTAL:
        print(f"[note] came in {TOTAL - len(selected)} short of target -- capacity exhausted "
              f"across all 6 pools combined, not just mapiq. See per-domain breakdown above.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "raw_grpo5k_records.jsonl"
    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in selected) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(selected)} raw records to {out_path}")


if __name__ == "__main__":
    main()
