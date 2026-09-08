#!/usr/bin/env python3
"""
Walk the ORIGINAL split_reviewed-2/{Train,Val,Test}/<domain> pools and write
one raw adapter-contract JSONL per (domain, split), with a single UNIFORM
filter policy applied everywhere -- no per-stage divergence, and independent
of which model is eventually trained on the output.

Filters (the only three applied, at every stage):
  1. drop if there is no bbox annotation at all (nothing to train/score on)
  2. drop if the referenced image file is missing on disk
  3. SANITIZE (not drop) the malformed MapIQ `choices` field -- upstream
     bug where some MapIQ choices lists are stringified-dict fragments
     (containing "{", "'", or ":"); cleaned to an empty list, since a
     malformed field is a data bug, not a property of the sample worth
     excluding it over.

Everything else a curated pipeline might filter -- dense (>15-box) targets,
full-image boxes, contradictory (image,question) duplicate gold sets -- is
INTENTIONALLY KEPT at every stage, so SFT, GRPO, validation, and test all
see the same class of examples and no comparison across stages is distorted
by curation. Contradictory duplicates are not a training crash risk under
this pipeline's plain JSON-array target: two differently-labeled copies of
the same (image, question) are just two ordinary training examples, the
same way any other label noise is.

No per-domain cap: every usable file in each split is kept. Domain-balanced
quota allocation and the SFT/GRPO split happen downstream in
build_dragon6_pools.py.

Usage:
    python3 build_dragon6_raw.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "split_reviewed-2"
IMAGE_ROOT = ROOT / "Diagram_Attribution_Dataset"
OUT_DIR = ROOT / "dragon_datasets" / "raw_by_split"

# short domain key -> split_reviewed-2 folder name
DOMAINS = {
    "ai2d": "ai2d",
    "chartqa": "ChartQA",
    "circuitvqa": "Circuit-VQA",
    "infographics": "Infographics",
    "mapiq": "MapIQ",
    "mapwise": "Mapwise",
}
ANSWER_KEYS = ["gt_answer", "ground_truth_answer"]


def clean_choices(raw):
    """Drop lists that leak stringified-dict fragments (upstream MapIQ bug)
    rather than keep clearly-malformed text as if it were real options."""
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


def to_adapter_record(raw: dict, fallback_id: str) -> dict:
    if isinstance(raw.get("answers"), dict):
        answer = raw["answers"].get("correct", "")
        explanation_raw = raw["answers"].get("predicted_explanation", "")
    else:
        answer = next((raw[k] for k in ANSWER_KEYS if raw.get(k)), "")
        explanation_raw = raw.get("predicted_explanation", "")

    bbox = raw.get("bbox")
    if not isinstance(bbox, list) or not bbox:
        bbox = raw.get("boxes", [])

    return {
        "id": raw.get("q_id", fallback_id),
        "question": raw.get("question_text", ""),
        "choices": clean_choices(raw.get("choices")),
        "answer": str(answer).strip(),
        "image_path": raw["image_path"],
        "bbox": bbox,
        "explanation_raw": explanation_raw,
    }


def walk_domain_split(dom_key: str, folder_name: str, split: str) -> list:
    src = SPLIT_DIR / split / folder_name
    if not src.exists():
        print(f"  [warn] {src} does not exist, skipping")
        return []
    records, n_no_box, n_missing_image = [], 0, 0
    for i, fp in enumerate(sorted(src.glob("*.json"))):
        raw = json.loads(fp.read_text())
        if not has_boxes(raw):
            n_no_box += 1
            continue
        rec = to_adapter_record(raw, fallback_id=f"{folder_name}-{split}-{i}")
        if not (IMAGE_ROOT / rec["image_path"]).exists():
            n_missing_image += 1
            continue
        rec["id"] = f"{dom_key}::{rec['id']}#i{i}"  # unique + traceable across the whole split
        records.append(rec)
    print(f"  [{dom_key}/{split}] kept {len(records)}  "
          f"(dropped no_bbox={n_no_box}, missing_image={n_missing_image})")
    return records


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    totals = {}
    for split in ("Train", "Val", "Test"):
        print(f"\n=== {split} ===")
        split_total = 0
        for dom_key, folder_name in DOMAINS.items():
            records = walk_domain_split(dom_key, folder_name, split)
            split_total += len(records)
            out_path = OUT_DIR / split / f"{dom_key}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                encoding="utf-8",
            )
        totals[split] = split_total
        print(f"  {split} total: {split_total}")

    print(f"\nWrote raw per-domain, per-split JSONL under {OUT_DIR}")
    print(f"Totals: {totals}")
    print("\nNext: python3 build_dragon6_pools.py         (SFT/GRPO split from Train)")
    print("      python3 build_dragon6_val.py             (domain-interleaved Val)")
    print("      python3 ../dragon_grpo/build_test_split.py  (Test, per-domain)")


if __name__ == "__main__":
    main()
