#!/usr/bin/env python3
"""
Convert a raw adapter-contract JSONL (id, question, choices, answer,
image_path, bbox) into this repo's stored conversation schema:
    {id, image, conversations: [{from:system,value}, {from:user,value},
     {from:assistant,value}], metadata: {answer, image_size, gt_boxes_norm}}

Model-agnostic: no chat templating happens here (that's model_config.py's
job, at train/inference time). This step only needs box geometry and the
shared prompt text, so its output is reused unchanged no matter which
model_config a later training/eval run points at.

metadata.gt_boxes_norm is what the GRPO reward and eval export read --
carrying it alongside the training target (rather than recomputing it from
the raw bbox at reward time) guarantees training target and reward can never
silently disagree about the coordinate space.

Usage:
    python3 convert_to_chat_jsonl.py \
        --in dragon_datasets/raw_sft_records.jsonl \
        --out dragon_datasets/sft_grounding.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from box_geometry import normalize_and_sort_gt_boxes
from grounding_prompts import GROUNDING_SYSTEM_PROMPT, build_target, build_user_prompt

ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = ROOT / "Diagram_Attribution_Dataset"


def convert_record(rec: dict, image_root: Path) -> dict | None:
    image_rel = rec.get("image_path") or rec.get("image")
    image_abs = (image_root / image_rel).resolve()
    if not image_abs.exists():
        return None
    from PIL import Image
    with Image.open(image_abs) as img:
        image_size = img.size

    boxes = normalize_and_sort_gt_boxes(rec.get("bbox") or [], image_size)
    if not boxes:
        return None

    return {
        "id": rec.get("id"),
        "image": image_rel,
        "conversations": [
            {"from": "system", "value": GROUNDING_SYSTEM_PROMPT},
            {"from": "user", "value": build_user_prompt(
                question=rec.get("question", ""),
                choices=rec.get("choices") or [],
                answer=rec.get("answer", ""))},
            {"from": "assistant", "value": build_target(boxes)},
        ],
        "metadata": {
            "answer": rec.get("answer", ""),
            "image_size": list(image_size),
            "gt_boxes_norm": [list(b) for b in boxes],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-root", default=str(IMAGE_ROOT))
    args = ap.parse_args()

    image_root = Path(args.image_root)
    records = [json.loads(l) for l in Path(args.inp).read_text().splitlines() if l.strip()]

    converted, skipped = [], 0
    for rec in records:
        out = convert_record(rec, image_root)
        if out is None:
            skipped += 1
            continue
        converted.append(out)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in converted) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(converted)} samples -> {out_path} (skipped {skipped}: "
          f"missing image or no usable gt boxes)")


if __name__ == "__main__":
    main()
